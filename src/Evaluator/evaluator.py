"""
Model Evaluator module for Autoencoder and Hybrid (Centroid-based) anomaly detection models.
Provides single and joint side-by-side performance evaluation.
"""

import logging
import time
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score
from Model.Centroid import CentroidBasedOneClassClassifier

# Configure the logging module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class Evaluator(object):

    def __init__(
        self,
        model,
        model_type="both",
        metric="AUC",
        device=None
    ) -> None:

        """
        :param model: Trained PyTorch model (Autoencoder or similar)
        :param model_type: "autoencoder", "hybrid", or "both" for direct comparison
        :param metric: "AUC", "classification", or "time"
        :param device: torch.device instance
        """

        self.device = (
            device
            if device is not None
            else torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.model = model.to(self.device)
        self.model_type = model_type.lower()
        self.metric = metric

    # ============================================================
    # EXTRACT FEATURES AND RECONSTRUCTION
    # ============================================================

    def _extract_features_and_output(self, batch_data):

        """
        Safely extracts latent representation and reconstructed
        output from the model.

        VAE returns:

            (output, mu, logvar)

        Therefore:

            output = model_out[0]  -> reconstruction X_hat
            latent = model_out[1]  -> mu

        The previous implementation assumed:

            (latent, reconstruction)

        which was incorrect for the current VAE.
        """

        model_out = self.model(batch_data)

        # --------------------------------------------------------
        # VAE / Autoencoder returning tuple
        #
        # Current VAE:
        #
        #     (output, mu, logvar)
        #
        # --------------------------------------------------------

        if isinstance(model_out, (tuple, list)):

            # First element is the reconstructed output
            output = model_out[0]

            # Second element is mu
            # which is used as the deterministic latent
            # representation for Hybrid evaluation.
            if len(model_out) > 1:
                latent = model_out[1]
            else:
                latent = output

        else:

            # Standard model returning only reconstruction
            output = model_out

            # ----------------------------------------------------
            # If model has encode(), extract latent representation
            # ----------------------------------------------------

            if hasattr(self.model, "encode"):

                encoded = self.model.encode(
                    batch_data
                )

                # VAE encode() returns:
                #
                #     (mu, logvar)
                #
                # Use mu as latent representation.
                if isinstance(
                    encoded,
                    (tuple, list)
                ):
                    latent = encoded[0]
                else:
                    latent = encoded

            else:

                latent = output

        return latent, output

    # ============================================================
    # AUC
    # ============================================================

    def calculate_auc(self, y_true, score):

        if not np.all(np.isfinite(score)):

            logging.warning(
                "Non-finite anomaly scores detected. "
                "Replacing NaN/Inf values."
            )

            score = np.nan_to_num(score)

        fpr, tpr, _ = roc_curve(
            y_true,
            score
        )

        return auc(
            fpr,
            tpr
        )

    # ============================================================
    # SCORE TO LABEL
    # ============================================================

    def score_to_label(
        self,
        score,
        threshold=0.5
    ):

        return np.where(
            np.array(score) > threshold,
            1,
            0
        )

    # ============================================================
    # CLASSIFICATION METRICS
    # ============================================================

    def calculate_classification_metrics(
        self,
        y_true,
        score,
        threshold=0.5
    ):

        y_pred = self.score_to_label(
            score,
            threshold
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        return (
            f1,
            precision,
            recall
        )

    # ============================================================
    # AUTOENCODER / VAE EVALUATION
    # ============================================================

    def _eval_autoencoder(self, test_loader):

        """
        Evaluates model using pure Reconstruction Error (MSE).

        For the current VAE:

            X
            ↓
          Encoder
            ↓
        mu, logvar
            ↓
       reparameterization
            ↓
            z
            ↓
         Decoder
            ↓
           X_hat

        Anomaly score:

            MSE(X, X_hat)

        IMPORTANT:
        The VAE's mu is NOT used as the reconstruction.
        """

        anomaly_scores = []
        test_labels = []

        with torch.no_grad():

            for batch_input in tqdm(
                test_loader,
                desc='Evaluating Autoencoder...'
            ):

                batch_data = (
                    batch_input[0]
                    .to(self.device)
                )

                # ------------------------------------------------
                # Extract:
                #
                # latent = mu
                # output = reconstruction X_hat
                # ------------------------------------------------

                _, output = (
                    self._extract_features_and_output(
                        batch_data
                    )
                )

                # ------------------------------------------------
                # Reconstruction MSE per sample
                #
                # MSE(X, X_hat)
                # ------------------------------------------------

                recon_loss = torch.mean(
                    torch.nn.MSELoss(
                        reduction="none"
                    )(
                        batch_data,
                        output
                    ),
                    dim=1
                )

                anomaly_scores.append(
                    recon_loss
                )

                test_labels.append(
                    batch_input[1]
                )

        # ========================================================
        # Combine all batches
        # ========================================================

        anomaly_scores = torch.cat(
            anomaly_scores,
            dim=0
        ).cpu().numpy()

        test_labels = torch.cat(
            test_labels,
            dim=0
        ).cpu().numpy()

        # ========================================================
        # Calculate AUC
        # ========================================================

        auc_val = self.calculate_auc(
            test_labels,
            anomaly_scores
        )

        # ========================================================
        # Calculate F1 / Precision / Recall
        #
        # Existing threshold = 0.5
        # ========================================================

        (
            f1_val,
            prec_val,
            rec_val
        ) = self.calculate_classification_metrics(
            test_labels,
            anomaly_scores
        )

        return {
            "scores": anomaly_scores,
            "labels": test_labels,
            "auc": auc_val,
            "f1": f1_val,
            "precision": prec_val,
            "recall": rec_val
        }

    # ============================================================
    # HYBRID EVALUATION
    # ============================================================

    def _eval_hybrid(
        self,
        train_loader,
        test_loader
    ):

        """
        Evaluates model using Latent Representation
        + Centroid Classifier.
        """

        if train_loader is None:

            raise ValueError(
                "🚨 Error: 'train_loader' is required "
                "for Hybrid evaluation to fit Centroid classifier."
            )

        train_latents = []
        test_latents = []
        test_labels = []

        with torch.no_grad():

            # ====================================================
            # 1. Extract Train Latent Space
            # ====================================================

            for batch_input in tqdm(
                train_loader,
                desc='Extracting Train Latents...'
            ):

                batch_data = (
                    batch_input[0]
                    .to(self.device)
                )

                # For VAE:
                #
                # latent = mu
                # output = reconstruction

                latent, _ = (
                    self._extract_features_and_output(
                        batch_data
                    )
                )

                train_latents.append(
                    latent
                )

            # ====================================================
            # 2. Extract Test Latent Space
            # ====================================================

            for batch_input in tqdm(
                test_loader,
                desc='Extracting Test Latents...'
            ):

                batch_data = (
                    batch_input[0]
                    .to(self.device)
                )

                # For VAE:
                #
                # latent = mu

                latent, _ = (
                    self._extract_features_and_output(
                        batch_data
                    )
                )

                test_latents.append(
                    latent
                )

                test_labels.append(
                    batch_input[1]
                )

        # ========================================================
        # Convert to NumPy
        # ========================================================

        train_latents = torch.cat(
            train_latents,
            dim=0
        ).cpu().numpy()

        test_latents = torch.cat(
            test_latents,
            dim=0
        ).cpu().numpy()

        test_labels = torch.cat(
            test_labels,
            dim=0
        ).cpu().numpy()

        # ========================================================
        # Fit Centroid Model
        # ========================================================

        cen = CentroidBasedOneClassClassifier()

        cen.fit(
            train_latents
        )

        # ========================================================
        # Centroid inference
        # ========================================================

        start_time = time.time()

        predictions_cen = cen.get_density(
            test_latents
        )

        infer_time = (
            time.time()
            - start_time
        )

        # ========================================================
        # Metrics
        # ========================================================

        auc_val = self.calculate_auc(
            test_labels,
            predictions_cen
        )

        (
            f1_val,
            prec_val,
            rec_val
        ) = self.calculate_classification_metrics(
            test_labels,
            predictions_cen
        )

        return {
            "scores": predictions_cen,
            "labels": test_labels,
            "train_latents": train_latents,
            "test_latents": test_latents,
            "auc": auc_val,
            "f1": f1_val,
            "precision": prec_val,
            "recall": rec_val,
            "infer_time": infer_time
        }

    # ============================================================
    # MAIN EVALUATION CONTROLLER
    # ============================================================

    def evaluate(
        self,
        test_loader,
        train_loader=None
    ):

        """
        Main evaluation controller.

        Supports:

            "autoencoder"
            "hybrid"
            "both"
        """

        self.model.eval()

        results = {}

        # ========================================================
        # Autoencoder / VAE
        # ========================================================

        if self.model_type in [
            "autoencoder",
            "both"
        ]:

            results["autoencoder"] = (
                self._eval_autoencoder(
                    test_loader
                )
            )

        # ========================================================
        # Hybrid
        # ========================================================

        if self.model_type in [
            "hybrid",
            "both"
        ]:

            results["hybrid"] = (
                self._eval_hybrid(
                    train_loader,
                    test_loader
                )
            )

        # ========================================================
        # Print comparison
        # ========================================================

        self._log_comparison(
            results
        )

        # ========================================================
        # Return requested metric
        # ========================================================

        if self.model_type == "autoencoder":

            return (
                results["autoencoder"]["auc"]
                if self.metric == "AUC"
                else results["autoencoder"]["f1"]
            )

        elif self.model_type == "hybrid":

            if self.metric == "time":

                return results["hybrid"]["infer_time"]

            return (
                results["hybrid"]["auc"]
                if self.metric == "AUC"
                else results["hybrid"]["f1"]
            )

        else:

            return results

    # ============================================================
    # COMPARISON LOG
    # ============================================================

    def _log_comparison(
        self,
        results
    ):

        """
        Helper to log a neat comparison table in the console.
        """

        logging.info(
            "=" * 65
        )

        logging.info(
            f"{'METRIC SUMMARY COMPARISON':^65}"
        )

        logging.info(
            "=" * 65
        )

        has_ae = (
            "autoencoder"
            in results
        )

        has_hy = (
            "hybrid"
            in results
        )

        header = (
            f"{'Metric':<15} |"
        )

        if has_ae:

            header += (
                f" {'Autoencoder (MSE)':<20} |"
            )

        if has_hy:

            header += (
                f" {'Hybrid (Centroid)':<20} |"
            )

        logging.info(
            header
        )

        logging.info(
            "-" * 65
        )

        for metric_key, name in [
            ("auc", "AUC Score"),
            ("f1", "F1 Score"),
            ("precision", "Precision"),
            ("recall", "Recall")
        ]:

            row = (
                f"{name:<15} |"
            )

            if has_ae:

                row += (
                    f" {results['autoencoder'][metric_key]:<20.4f} |"
                )

            if has_hy:

                row += (
                    f" {results['hybrid'][metric_key]:<20.4f} |"
                )

            logging.info(
                row
            )

        logging.info(
            "=" * 65
        )

    # ============================================================
    # VISUALIZATION
    # ============================================================

    def visualize(self):
        pass
