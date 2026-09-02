import os
import argparse
import logging
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Correct Imports matching dataloader.py
from DataLoader.dataloader import load_data, IoTDataProccessor, IoTDataset
from Shrink_Autoencoder import Shrink_Autoencoder
from client_trainer import ClientTrainer
from global_aggregator import GlobalAggregator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Shrink Autoencoder Training")
    parser.add_argument("--base_data_path", type=str, required=True, help="Base path to client folders")
    parser.add_argument("--dev_data_path", type=str, required=True, help="Path to dev evaluation CSV folder")
    parser.add_argument("--num_clients", type=int, default=5, help="Number of federated clients")
    parser.add_argument("--num_rounds", type=int, default=20, help="Number of FL communication rounds")
    parser.add_argument("--local_epochs", type=int, default=5, help="Local training epochs per client")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    
    # Shrink Autoencoder Parameters
    parser.add_argument("--shrink_dim", type=int, default=16, help="Bottleneck/latent dimension")
    parser.add_argument("--shrink_threshold", type=float, default=0.2, help="Shrinkage operator threshold")

    parser.add_argument("--algorithm", type=str, default="fedavg", choices=["fedavg", "fedprox"], help="FL algorithm")
    parser.add_argument("--fedprox_mu", type=float, default=0.01, help="Mu parameter for FedProx")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save models")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Initialize Processor and Global Model Setup
    data_processor = IoTDataProccessor(scaler="standard")
    
    # 2. Process Client Data
    clients = []
    input_dim = None

    logging.info("Loading client datasets...")
    for i in range(1, args.num_clients + 1):
        client_folder = os.path.join(args.base_data_path, f"client_{i}")
        raw_df = load_data(client_folder)
        
        # Fit scaler on first client or transform across clients
        if i == 1:
            scaled_data, labels = data_processor.fit_transform(raw_df)
            input_dim = scaled_data.shape[1]
        else:
            scaled_data, labels = data_processor.transform(raw_df, type="normal")

        # Wrap with IoTDataset
        client_dataset = IoTDataset(scaled_data, labels)
        client_loader = DataLoader(client_dataset, batch_size=args.batch_size, shuffle=True)

        client_save_dir = os.path.join(args.save_dir, f"client_{i}")
        client_model = Shrink_Autoencoder(
            input_dim=input_dim, 
            shrink_dim=args.shrink_dim, 
            threshold=args.shrink_threshold
        )
        
        client = ClientTrainer(
            client_id=i,
            model=client_model,
            train_loader=client_loader,
            epochs=args.local_epochs,
            lr=args.lr,
            algorithm=args.algorithm,
            fedprox_mu=args.fedprox_mu,
            device=args.device,
            save_dir=client_save_dir
        )
        clients.append(client)

    # 3. Process Global Dev Data using fitted scaler
    logging.info("Loading global evaluation dev set...")
    dev_raw_df = load_data(args.dev_data_path)
    dev_scaled_data, dev_labels = data_processor.transform(dev_raw_df, type="normal")
    dev_dataset = IoTDataset(dev_scaled_data, dev_labels)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    # 4. Initialize Global Aggregator
    global_model = Shrink_Autoencoder(
        input_dim=input_dim, 
        shrink_dim=args.shrink_dim, 
        threshold=args.shrink_threshold
    ).to(args.device)

    global_aggregator = GlobalAggregator(global_model=global_model, dev_loader=dev_loader, device=args.device)

    # 5. Federated Training Loop
    logging.info("Starting Federated Training loop...")
    for round_num in range(1, args.num_rounds + 1):
        logging.info(f"--- Round {round_num}/{args.num_rounds} ---")
        client_weights = []
        client_data_sizes = []

        global_params = global_aggregator.get_global_parameters()

        for client in clients:
            client.set_parameters(global_params)
            loss = client.train()
            client_weights.append(client.get_parameters())
            client_data_sizes.append(len(client.train_loader.dataset))

        global_aggregator.aggregate(client_weights, client_data_sizes)

        dev_loss = global_aggregator.evaluate()
        logging.info(f"Round {round_num} Global Dev Loss: {dev_loss:.6f}")

    # Save final model
    global_save_path = os.path.join(args.save_dir, "global_shrink_autoencoder.pt")
    torch.save(global_model.state_dict(), global_save_path)
    logging.info(f"Training Complete. Saved global model to {global_save_path}")


if __name__ == "__main__":
    main()
