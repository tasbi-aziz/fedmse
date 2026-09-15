"""
PyTorch dataloader for training and evaluating models.
Updated for purely numerical datasets using Min-Max Scaling (0 to 1).
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from torch.utils.data import DataLoader, Dataset

# Configure logging module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(path, header=None):
    dataframe = []
    
    # --- COLAB FALLBACK PATHS ---
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


class VarianceKBestSelector:
    """Helper selector to choose Top-K features based on variance when only normal data is present."""
    def __init__(self, k):
        self.k = k
        self.selected_indices = None

    def fit(self, X):
        variances = np.var(X, axis=0)
        self.selected_indices = np.argsort(variances)[-self.k:]
        return self

    def transform(self, X):
        if self.selected_indices is None:
            return X
        return X[:, self.selected_indices]


class IoTDataProcessor(object):
    def __init__(self, scaler="minmax", use_log_transform=False, n_selected_features=None, threshold=0.01):
        """
        Default scaler set to 'minmax' [0, 1] for purely numerical data.
        """
        self.scaler_type = scaler
        self.use_log_transform = use_log_transform
        self.n_selected_features = n_selected_features
        self.threshold = threshold
        
        if scaler == "minmax":
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif scaler == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler}. Use 'minmax' or 'standard'.")

        self.selector = None

    def _to_numpy(self, dataframe):
        """Converts numerical dataframe or array directly to float32 numpy array."""
        if isinstance(dataframe, pd.DataFrame):
            return dataframe.values.astype(np.float32)
        return np.array(dataframe, dtype=np.float32)

    def _apply_log_transform(self, dataframe):
        """Applies log(1 + x) transformation if enabled."""
        values = self._to_numpy(dataframe)
        if not self.use_log_transform:
            return values

        clipped_values = np.maximum(0, values)
        return np.log1p(clipped_values)

    def fit_transform(self, dataframe, abnormal_dataframe=None):
        """Fits MinMaxScaler and Feature Selector on normal/abnormal client training data."""
        transformed_input = self._apply_log_transform(dataframe)
        
        # 1. MinMaxScaler Fit & Transform
        processed_data = self.scaler.fit_transform(transformed_input)
        
        # 2. Feature Selector Fit
        if self.n_selected_features is not None:
            total_features = processed_data.shape[1]
            actual_k = min(self.n_selected_features, total_features)

            if abnormal_dataframe is not None and len(abnormal_dataframe) > 0:
                trans_abnormal = self._apply_log_transform(abnormal_dataframe)
                proc_abnormal = self.scaler.transform(trans_abnormal)
                
                x_sample = np.vstack([processed_data, proc_abnormal])
                y_sample = np.hstack([np.zeros(len(processed_data)), np.ones(len(proc_abnormal))])
                
                self.selector = SelectKBest(score_func=f_classif, k=actual_k)
                self.selector.fit(x_sample, y_sample)
            else:
                self.selector = VarianceKBestSelector(k=actual_k)
                self.selector.fit(processed_data)
        else:
            self.selector = VarianceThreshold(threshold=self.threshold)
            self.selector.fit(processed_data)

        # 3. Apply Feature Selection
        processed_data = self.selector.transform(processed_data)
        
        label = np.zeros(len(processed_data), dtype=np.float32)
        return processed_data, label

    def transform(self, dataframe, type="normal"):
        """Transforms validation or test data using fitted MinMaxScaler and Selector."""
        transformed_input = self._apply_log_transform(dataframe)
        processed_data = self.scaler.transform(transformed_input)
        
        if self.selector is not None:
            processed_data = self.selector.transform(processed_data)
        
        if type == "normal":
            label = np.zeros(len(dataframe), dtype=np.float32)
        else:
            label = np.ones(len(dataframe), dtype=np.float32)
            
        return processed_data, label
    
    def get_metadata(self):
        if isinstance(self.scaler, MinMaxScaler):
            metadata = {
                "min": self.scaler.data_min_,
                "max": self.scaler.data_max_
            }
        elif isinstance(self.scaler, StandardScaler):
            metadata = {
                "mean": self.scaler.mean_,
                "std": self.scaler.scale_
            }
        else:
            metadata = {}
        return metadata


class IoTDataset(Dataset):
    """Custom PyTorch Dataset class for N-BAIoT data."""
    
    def __init__(self, data, label):
        self.data = data.astype(np.float32)
        self.label = label.astype(np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.label[idx], dtype=torch.float32)
        return X, y
    
    @property
    def input_dim_(self):
        return self.data.shape[1]


# Helper Alias for backward compatibility
IoTDataProccessor = IoTDataProcessor
