"""
PyTorch dataloader for training and evaluating models.

Preprocessing pipeline:
1. Log1p transformation on all original numerical features
2. Unsupervised feature selection
3. Min-Max Scaling to [0, 1]
4. PyTorch Dataset conversion

Designed for N-BaIoT numerical, normal-only / unsupervised training.
"""

import os
import logging
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from torch.utils.data import DataLoader, Dataset


# Configure logging module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_data(path, header=None):
    dataframe = []

    # --- COLAB FALLBACK PATHS ---
    if not os.path.exists(path):
        logging.warning(
            f"Path {path} not found. Redirecting to absolute Colab directory..."
        )

        folder_name = os.path.basename(os.path.normpath(path))
        colab_fallback_path = (
            f"/content/fedmse/Data/noniid-10-Client_Data/{folder_name}"
        )

        if os.path.exists(colab_fallback_path):
            path = colab_fallback_path
            logging.info(
                f"Successfully redirected to absolute path: {path}"
            )
        else:
            alternative_path = f"/content/fedmse/Data/{folder_name}"

            if os.path.exists(alternative_path):
                path = alternative_path
                logging.info(
                    f"Successfully redirected to alternative absolute path: {path}"
                )

    if not os.path.exists(path) or not os.listdir(path):
        raise FileNotFoundError(
            f"🚨 Error: Data folder could not be found or is empty at: {path}"
        )

    for file in os.listdir(path):
        if file.endswith(".csv"):
            filename = os.path.join(path, file)

            logging.info(f"Loading {filename}")

            dataframe.append(
                pd.read_csv(filename, header=header)
            )

    if not dataframe:
        raise ValueError(
            f"🚨 Error: No CSV files found inside the directory: {path}"
        )

    dataframe = pd.concat(
        dataframe,
        ignore_index=True
    )

    return dataframe


class VarianceKBestSelector:
    """
    Unsupervised selector to choose the Top-K features
    based on variance.

    This does not use labels or abnormal data.
    """

    def __init__(self, k):
        self.k = k
        self.selected_indices = None

    def fit(self, X):
        variances = np.var(X, axis=0)

        # Select the indices of the K highest-variance features.
        self.selected_indices = np.argsort(variances)[-self.k:]

        return self

    def transform(self, X):
        if self.selected_indices is None:
            return X

        return X[:, self.selected_indices]

    def get_support(self):
        if self.selected_indices is None:
            return None

        return self.selected_indices


class IoTDataProcessor(object):

    def __init__(
        self,
        scaler="minmax",
        use_log_transform=False,
        n_selected_features=None,
        threshold=0.01
    ):
        """
        Data preprocessing pipeline:

        Original Features
            ↓
        Log Transformation
            ↓
        Feature Selection
            ↓
        Scaling
            ↓
        Model Input

        Parameters
        ----------
        scaler : str
            'minmax' or 'standard'

        use_log_transform : bool
            Whether to apply log1p transformation.

        n_selected_features : int or None
            Number of features to keep.
            Example:
                115 → 50
                115 → 30
                115 → 20

        threshold : float
            Variance threshold used when
            n_selected_features is None.
        """

        self.scaler_type = scaler
        self.use_log_transform = use_log_transform
        self.n_selected_features = n_selected_features
        self.threshold = threshold

        # Scaler is fitted AFTER feature selection.
        if scaler == "minmax":
            self.scaler = MinMaxScaler(feature_range=(0, 1))

        elif scaler == "standard":
            self.scaler = StandardScaler()

        else:
            raise ValueError(
                f"Unknown scaler type: {scaler}. "
                f"Use 'minmax' or 'standard'."
            )

        self.selector = None

    def _to_numpy(self, dataframe):
        """
        Converts a numerical dataframe or array
        directly to float32 NumPy array.
        """

        if isinstance(dataframe, pd.DataFrame):
            return dataframe.values.astype(np.float32)

        return np.array(
            dataframe,
            dtype=np.float32
        )

    def _apply_log_transform(self, dataframe):
        """
        Applies log1p(x) transformation.

        For non-negative N-BaIoT numerical features:
            log1p(x) = log(1 + x)

        If transformation is disabled,
        original numerical values are returned.
        """

        values = self._to_numpy(dataframe)

        if not self.use_log_transform:
            return values

        # N-BaIoT numerical traffic features are expected
        # to be non-negative.
        clipped_values = np.maximum(0, values)

        return np.log1p(clipped_values).astype(np.float32)

    def fit_transform(self, dataframe, abnormal_dataframe=None):
        """
        Fits the preprocessing pipeline on training data.

        IMPORTANT:
        Feature selection is completely UNSUPERVISED.

        abnormal_dataframe is retained in the function signature
        only for backward compatibility. It is NOT used for
        feature selection.

        Pipeline:
            Original 115 features
                    ↓
            Log transformation
                    ↓
            Unsupervised feature selection
                    ↓
            MinMax/Standard scaling
                    ↓
            Selected feature matrix X
        """

        # ---------------------------------------------------------
        # 1. Log Transformation
        # ---------------------------------------------------------
        transformed_input = self._apply_log_transform(dataframe)

        original_feature_count = transformed_input.shape[1]

        logging.info(
            f"Original feature count: {original_feature_count}"
        )

        # ---------------------------------------------------------
        # 2. UNSUPERVISED FEATURE SELECTION
        # ---------------------------------------------------------
        if self.n_selected_features is not None:

            total_features = transformed_input.shape[1]

            actual_k = min(
                self.n_selected_features,
                total_features
            )

            self.selector = VarianceKBestSelector(
                k=actual_k
            )

            self.selector.fit(
                transformed_input
            )

            processed_data = self.selector.transform(
                transformed_input
            )

            logging.info(
                f"Feature selection: "
                f"{original_feature_count} → "
                f"{processed_data.shape[1]} features"
            )

        else:
            # Remove features with variance below threshold.
            self.selector = VarianceThreshold(
                threshold=self.threshold
            )

            processed_data = self.selector.fit_transform(
                transformed_input
            )

            logging.info(
                f"VarianceThreshold selection: "
                f"{original_feature_count} → "
                f"{processed_data.shape[1]} features"
            )

        # ---------------------------------------------------------
        # 3. SCALING
        # ---------------------------------------------------------
        processed_data = self.scaler.fit_transform(
            processed_data
        ).astype(np.float32)

        logging.info(
            f"Scaling complete using {self.scaler_type}. "
            f"Final input dimension: {processed_data.shape[1]}"
        )

        # ---------------------------------------------------------
        # 4. Training labels
        # ---------------------------------------------------------
        # Training is normal-only / unsupervised.
        label = np.zeros(
            len(processed_data),
            dtype=np.float32
        )

        return processed_data, label

    def transform(self, dataframe, type="normal"):
        """
        Applies the already-fitted preprocessing pipeline
        to validation or test data.

        IMPORTANT:
        No fitting happens here.

        The same:
            Log transformation
            Feature selection
            Scaler

        learned from training data are reused.
        """

        # ---------------------------------------------------------
        # 1. Log Transformation
        # ---------------------------------------------------------
        transformed_input = self._apply_log_transform(
            dataframe
        )

        # ---------------------------------------------------------
        # 2. Feature Selection
        # ---------------------------------------------------------
        if self.selector is not None:

            processed_data = self.selector.transform(
                transformed_input
            )

        else:
            processed_data = transformed_input

        # ---------------------------------------------------------
        # 3. Scaling
        # ---------------------------------------------------------
        processed_data = self.scaler.transform(
            processed_data
        ).astype(np.float32)

        # ---------------------------------------------------------
        # 4. Labels
        # ---------------------------------------------------------
        if type == "normal":
            label = np.zeros(
                len(dataframe),
                dtype=np.float32
            )

        else:
            label = np.ones(
                len(dataframe),
                dtype=np.float32
            )

        return processed_data, label

    def get_selected_features(self):
        """
        Returns the indices of the selected original features.
        """

        if self.selector is None:
            return None

        if isinstance(
            self.selector,
            VarianceKBestSelector
        ):
            return self.selector.get_support()

        if hasattr(
            self.selector,
            "get_support"
        ):
            return self.selector.get_support(
                indices=True
            )

        return None

    def get_metadata(self):
        """
        Returns scaling metadata.
        """

        if isinstance(
            self.scaler,
            MinMaxScaler
        ):

            metadata = {
                "min": self.scaler.data_min_,
                "max": self.scaler.data_max_
            }

        elif isinstance(
            self.scaler,
            StandardScaler
        ):

            metadata = {
                "mean": self.scaler.mean_,
                "std": self.scaler.scale_
            }

        else:
            metadata = {}

        return metadata


class IoTDataset(Dataset):
    """
    Custom PyTorch Dataset class for N-BaIoT data.
    """

    def __init__(self, data, label):

        self.data = data.astype(
            np.float32
        )

        self.label = label.astype(
            np.float32
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        X = torch.tensor(
            self.data[idx],
            dtype=torch.float32
        )

        y = torch.tensor(
            self.label[idx],
            dtype=torch.float32
        )

        return X, y

    @property
    def input_dim_(self):
        return self.data.shape[1]


# Helper Alias for backward compatibility
IoTDataProccessor = IoTDataProcessor
