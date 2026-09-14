"""
PyTorch dataloader for training and evaluating models.
Updated with safe feature selection bounds and character cleaning.
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from torch.utils.data import DataLoader, Dataset

# Configure logging module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
        if file.endswith(".csv"):
            filename = os.path.join(path, file)
            logging.info(f"Loading {filename}")
            dataframe.append(pd.read_csv(filename, header=header))
            
    if not dataframe:
        raise ValueError(f"🚨 Error: No CSV files found inside the directory: {path}")
        
    dataframe = pd.concat(dataframe, ignore_index=True)
    return dataframe


class IoTDataProccessor(object):
    def __init__(self, scaler="standard", use_log_transform=True, n_selected_features=None, threshold=0.01):
        self.scaler_type = scaler
        self.use_log_transform = use_log_transform
        self.n_selected_features = n_selected_features
        self.threshold = threshold
        
        if scaler == "standard":
            self.scaler = StandardScaler()
        elif scaler == "minmax":
            self.scaler = MinMaxScaler((0, 1))
        else:
            raise ValueError(f"Unknown scaler type: {scaler}. Use 'standard' or 'minmax'.")

        self.selector = None

    def _apply_log_transform(self, dataframe):
        """Applies log(1 + x) transformation to smooth high-variance features."""
        if not self.use_log_transform:
            return dataframe

        if isinstance(dataframe, pd.DataFrame):
            values = dataframe.values
        else:
            values = np.array(dataframe)

        clipped_values = np.maximum(0, values)
        return np.log1p(clipped_values)

    def fit_transform(self, dataframe, abnormal_dataframe=None):
        """Fits Scaler and Feature Selector on normal/abnormal client training data."""
        transformed_input = self._apply_log_transform(dataframe)
        
        # 1. Scaler Fit & Transform
        processed_data = self.scaler.fit_transform(transformed_input)
        
        # 2. Feature Selector Fit
        if self.n_selected_features is not None:
            total_features = processed_data.shape[1]
            actual_k = min(self.n_selected_features, total_features)
            self.selector = SelectKBest(score_func=f_classif, k=actual_k)

            if abnormal_dataframe is not None:
                trans_abnormal = self._apply_log_transform(abnormal_dataframe)
                proc_abnormal = self.scaler.transform(trans_abnormal)
                
                x_sample = np.vstack([processed_data, proc_abnormal])
                y_sample = np.hstack([np.zeros(len(processed_data)), np.ones(len(proc_abnormal))])
                self.selector.fit(x_sample, y_sample)
            else:
                dummy_y = np.zeros(len(processed_data))
                self.selector.fit(processed_data, dummy_y)
        else:
            self.selector = VarianceThreshold(threshold=self.threshold)
            self.selector.fit(processed_data)

        # 3. Apply Feature Selection
        processed_data = self.selector.transform(processed_data)
        
        label = [0 for _ in range(len(dataframe))]
        return processed_data, np.array(label)

    def transform(self, dataframe, type="normal"):
        """Transforms validation or test data using fitted Scaler and Selector."""
        transformed_input = self._apply_log_transform(dataframe)
        processed_data = self.scaler.transform(transformed_input)
        
        if self.selector is not None:
            processed_data = self.selector.transform(processed_data)
        
        if type == "normal":
            label = [0 for _ in range(len(dataframe))]
        else:
            label = [1 for _ in range(len(dataframe))]
            
        return processed_data, np.array(label)
    
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
    """Custom PyTorch Dataset class for N-BAIoT data."""
    
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


# Alias for clean spelling compatibility
IoTDataProcessor = IoTDataProccessor
