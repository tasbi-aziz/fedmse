"""
This is the global model update module.
Handles multi-strategy model weight aggregation (FedAvg, FedProx, MSE-Avg, Fusion-Avg).
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
from Utils import similarity_score
from DataLoader import IoTDataProccessor

import logging

# Configure the logging module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GlobalAggregator(object):
    def __init__(self, model, update_type="avg"):
        """
        Initialize Global Aggregator.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.update_type = update_type
        self.model = model.to(self.device)
        self.dev_dataset = None
        self.dev_label = None
        self.val_loss = 0.0

    def create_dev_dataset(self, dataset):
        """
        Set up the development dataset for global model evaluation and weighted aggregation.
        """
        logging.info("Creating development dataset...")
        self.dev_dataset = dataset["dataset"]
        data_processor = IoTDataProccessor(scaler="standard")
        self.dev_dataset, self.dev_label = data_processor.fit_transform(self.dev_dataset)
        
        if self.update_type == "fusion_avg":
            self.dev_kde_scores = KernelDensity(kernel='gaussian', bandwidth="scott") \
                .fit(self.dev_dataset).score_samples(self.dev_dataset)

    def fusion_avg(self, local_models=None):
        """
        Perform KDE fusion-based updating by weighting models via data similarity.
        """
        logging.info("Fusion-based updating...")
        update_weights = []
        for i, local_model_entry in zip(tqdm(range(len(local_models)), desc='Calculating similarity...'), local_models):
            state_dict = local_model_entry[0]
            self.model.load_state_dict(state_dict)
            self.model.eval()
            with torch.no_grad():
                _, generated_data, _ = self.model(torch.Tensor(self.dev_dataset).to(self.device))
                sim_score = similarity_score(self.dev_kde_scores, generated_data.cpu().numpy())
                weight = 1.0 / (sim_score + 1e-8)
                update_weights.append((state_dict, weight))
                
        avg_weights = {}
        total_weight = sum([w for _, w in update_weights])
        for key in update_weights[0][0].keys():
            avg_weights[key] = sum([w[key] * alpha for w, alpha in update_weights]) / total_weight
        
        self.model.load_state_dict(avg_weights)

    def fed_mse_avg(self, local_models=None):
        """
        Perform MSE loss-based weight aggregation on AE models using validation data.
        """
        logging.info("MSE-based updating...")
        update_weights = []
        for i, local_model_entry in zip(tqdm(range(len(local_models)), desc='Calculating MSE...'), local_models):
            state_dict = local_model_entry[0]
            self.model.load_state_dict(state_dict)
            self.model.eval()
            with torch.no_grad():
                _, generated_data, _ = self.model(torch.Tensor(self.dev_dataset).to(self.device))
                mse_loss = torch.nn.MSELoss(reduction='mean')(torch.Tensor(self.dev_dataset).to(self.device), generated_data)
                sim_score = mse_loss.item()
                weight = 1.0 / (sim_score + 1e-8)
                update_weights.append((state_dict, weight))
                
        avg_weights = {}
        total_weight = sum([w for _, w in update_weights])
        for key in update_weights[0][0].keys():
            avg_weights[key] = sum([w[key] * alpha for w, alpha in update_weights]) / total_weight
                
        self.model.load_state_dict(avg_weights)

    def fed_avg(self, local_models=None):
        """
        Perform standard dataset size-weighted Federated Averaging (FedAvg).
        local_models structure: list of tuples (state_dict, total_samples, client_samples)
        """
        total_samples = sum(model[2] for model in local_models)
        avg_weights = {}

        for key in local_models[0][0].keys():
            avg_weights[key] = sum(model[0][key] * (model[2] / total_samples) for model in local_models)

        self.model.load_state_dict(avg_weights)

    def fedprox(self, local_models=None, mu=0.01):
        """
        Perform weighted aggregation using FedProx style.
        """
        logging.info("FedProx updating...")
        total_samples = sum(model[2] for model in local_models)
        avg_weights = {}

        for key in local_models[0][0].keys():
            avg_weights[key] = sum(model[0][key] * (model[2] / total_samples) for model in local_models)

        self.model.load_state_dict(avg_weights)

    def update(self, local_models=None):
        """
        Main entry point to update the global model with incoming local model lists.
        """
        if not local_models:
            logging.warning("No local models received for global update.")
            return

        if self.update_type == "avg":
            self.fed_avg(local_models)
        elif self.update_type == "fusion_avg":
            self.fusion_avg(local_models)
        elif self.update_type == "mse_avg":
            self.fed_mse_avg(local_models)
        elif self.update_type == "fedprox":
            self.fedprox(local_models, mu=0.01)

        # Compute validation loss on development dataset after aggregation
        self.model.eval()
        with torch.no_grad():
            _, _, val_loss = self.model(torch.Tensor(self.dev_dataset).to(self.device))
            self.val_loss = val_loss.item()
