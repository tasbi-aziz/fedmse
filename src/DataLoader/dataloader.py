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
    if not os.path.exists(path):
        logging.warning(f"Path {path} not found. Redirecting to absolute Colab directory...")
        
        folder_name = os.path.basename(os.path.normpath(path))
        colab_fallback_path = f"/content/fedmse/Data/noniid-10-Client_Data/{folder_name}"
        
        if os.path.exists(colab_fallback_path):
            path = colab_fallback_path
            logging.info(f"Successfully redirected to absolute path: {path}")
        else:
            alternative_path = f"/content/fedmse/Data/{folder_name}"
            if os.path.exists(alternative_path):
                path = alternative_path
                logging.info(f"Successfully redirected to alternative absolute path: {path}")

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
    def __init__(self, scaler="standard", use_log_transform=True):
        self.scaler_type = scaler
        self.use_log_transform = use_log_transform
        
        if scaler == "standard":
            self.scaler = StandardScaler()
        elif scaler == "minmax":
            self.scaler = MinMaxScaler((0, 1))
        else:
            raise ValueError(f"Unknown scaler type: {scaler}. Use 'standard' or 'minmax'.")

    def _apply_log_transform(self, dataframe):
        """Applies log(1 + x) transformation to smooth high-variance features."""
        if not self.use_log_transform:
            return dataframe

        if isinstance(dataframe, pd.DataFrame):
            values = dataframe.values
        else:
            values = np.array(dataframe)

        # Negative noise prevent korar jonno np.maximum(0, values) and log1p -> log(1 + x)
        clipped_values = np.maximum(0, values)
        return np.log1p(clipped_values)

    def transform(self, dataframe, type="normal"):
        transformed_input = self._apply_log_transform(dataframe)
        processed_data = self.scaler.transform(transformed_input)
        
        if type == "normal":
            label = [0 for _ in range(len(dataframe))]
        else:
            label = [1 for _ in range(len(dataframe))]
            
        return processed_data, np.array(label)
    
    def fit_transform(self, dataframe):
        transformed_input = self._apply_log_transform(dataframe)
        self.scaler = self.scaler.fit(transformed_input)
        processed_data, label = self.transform(dataframe=dataframe, type="normal")
        return processed_data, label
        
    def get_metadata(self):
        if isinstance(self.scaler, StandardScaler):
            metadata = {
                "mean": self.scaler.mean_,
                "std": self.scaler.scale_
            }
        elif isinstance(self.scaler, MinMaxScaler):
            metadata = {
                "min": self.scaler.data_min_,
                "max": self.scaler.data_max_
            }
        else:
            metadata = {}
        return metadata


class IoTDataset(Dataset):
    """
    A custom PyTorch Dataset class for the N-BAIoT dataset.
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
