"""
This is a PyTorch dataloader for training and evaluating a model.
@author
- Van Tuan Nguyen (vantuan.nguyen@lqdtu.edu.vn)
- Razvan Beuran (razvan@jaist.ac.jp)
@create date 2023-12-11 00:28:29
@modify date 2023-12-11 00:28:29
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, Dataset

import logging

# Configure the logging module
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(path, header=None):
    dataframe = []
    
    # --- PERMANENT COLAB FIX ---
    # If the provided path doesn't exist, automatically redirect to the absolute Colab path
    if not os.path.exists(path):
        logging.warning(f"Path {path} not found. Redirecting to absolute Colab directory...")
        
        # Extract the last part of the path (e.g., 'client_1') to look inside the unzipped dataset folder
        folder_name = os.path.basename(os.path.normpath(path))
        colab_fallback_path = f"/content/fedmse/Data/noniid-10-Client_Data/{folder_name}"
        
        if os.path.exists(colab_fallback_path):
            path = colab_fallback_path
            logging.info(f"Successfully redirected to absolute path: {path}")
        else:
            # If that fails, look into the base local data folder directly
            alternative_path = f"/content/fedmse/Data/{folder_name}"
            if os.path.exists(alternative_path):
                path = alternative_path
                logging.info(f"Successfully redirected to alternative absolute path: {path}")
    # ----------------------------

    if not os.path.exists(path) or not os.listdir(path):
        raise FileNotFoundError(f"🚨 Error: Data folder could not be found or is empty at: {path}")

    for file in os.listdir(path):
        if ".csv" in file:
            filename = os.path.join(path, file)
            logging.info(f"Loading {filename}")
            dataframe.append(pd.read_csv(filename, header=header))
            
    if not dataframe:
        raise ValueError(f"🚨 Error: No CSV files found inside the directory: {path}")
        
    dataframe = pd.concat(dataframe, ignore_index=True)
    return dataframe

class IoTDataProccessor(object):
    def __init__(self, scaler="standard"):
        if scaler == "standard":
            self.scaler = StandardScaler()
        
        if scaler == "minmax":
            self.scaler = MinMaxScaler((0, 1))

    def transform(self, dataframe, type="normal"):
        processed_data = self.scaler.transform(dataframe)
        if type == "normal":
            label = [0 for i in range(len(dataframe))]
        else:
            label = [1 for i in range(len(dataframe))]
        return processed_data, np.array(label)
    
    def fit_transform(self, dataframe):
        self.scaler = self.scaler.fit(dataframe)
        processed_data, label = self.transform(dataframe=dataframe, type="normal")
        return processed_data, label
        
    def get_metadata(self):
        metadata = {
            "mean": self.scaler.mean_,
            "std": self.scaler.scale_
        }
        return metadata
        
        
class IoTDataset(Dataset):
    """
    A custom Pytorch Dataset class for the N-BAIoT dataset.
    """
    
    def __init__(self, data, label):
        self.data = data
        self.label = label
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X = self.data[idx].astype(np.float32)
        y = self.label[idx].astype(np.float32)
        return X, y
    
    @property
    def input_dim_(self):
        return self.data.shape[1]
