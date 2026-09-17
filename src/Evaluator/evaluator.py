"""
Model Evaluator module for Autoencoder and Hybrid
(Centroid-based) anomaly detection models.

Supports:
1. Global combined evaluation
2. Client-wise evaluation
3. VAE reconstruction-MSE based anomaly detection
"""

import logging
import time
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score

from Model.Centroid import CentroidBasedOneClassClassifier


# Configure logging
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

        model_out = self.model(batch_data)

        # Current VAE:
        # (output, mu, logvar)

        if isinstance(model_out, (tuple, list)):

            output = model_out[0]

            if len(model_out) > 1:
                latent = model_out[1]
            else:
                latent = output

        else:

            output = model_out

            if hasattr(self.model, "encode"):

                encoded = self.model.encode(batch_data)

                if isinstance(encoded, (tuple, list)):
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

        anomaly_scores = []
        test_labels = []

        self.model.eval()

        with torch.no_grad():

            for batch_input in tqdm(
                test_loader,
                desc="Evaluating Autoencoder..."
            ):

                batch_data = (
                    batch_input[0]
                    .to(self.device)
                )

                _, output = (
                    self._extract_features_and_output(
                        batch_data
                    )
                )

                # Per-sample reconstruction MSE

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

        # --------------------------------------------------------
        # Combine batches
        # --------------------------------------------------------

        anomaly_scores = torch.cat(
            anomaly_scores,
            dim=0
        ).cpu().numpy()

        test_labels = torch.cat(
            test_labels,
            dim=0
        ).cpu().numpy()

        # --------------------------------------------------------
        # Metrics
        # --------------------------------------------------------

        auc_val = self.calculate_auc(
            test_labels,
            anomaly_scores
        )

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
    # CLIENT-WISE VAE EVALUATION
    # ============================================================

    def evaluate_client(
        self,
        client_id,
        test_loader
    ):

        """
        Evaluate the CURRENT GLOBAL MODEL on ONE client's
        test dataset.

        Each client's test_loader should contain:

            test_normal.csv  -> label 0
            abnormal.csv     -> label 1

        The model used here is the current global model.
        """

        self.model.eval()

        result = self._eval_autoencoder(
            test_loader
        )

        logging.info(
            f"[Client Evaluation] "
            f"{client_id} | "
            f"AUC: {result['auc']:.4f} | "
            f"F1: {result['f1']:.4f} | "
            f"Precision: {result['precision']:.4f} | "
            f"Recall: {result['recall']:.4f}"
        )

        return result

    # ============================================================
    # ALL CLIENTS EVALUATION
    # ============================================================

    def evaluate_all_clients(
        self,
        client_test_loaders
    ):

        """
        Evaluate the CURRENT GLOBAL MODEL on every client.

        client_test_loaders can be:

            {
                "Client-1": loader1,
                "Client-2": loader2,
                ...
            }

        Returns one result per client.
        """

        client_results = {}

        self.model.eval()

        logging.info(
            "=" * 70
        )

        logging.info(
            "CLIENT-WISE GLOBAL MODEL EVALUATION"
        )

        logging.info(
            "=" * 70
        )

        for client_id, test_loader in client_test_loaders.items():

            result = self.evaluate_client(
                client_id,
                test_loader
            )

            client_results[client_id] = result

        # --------------------------------------------------------
        # Client AUC summary
        # --------------------------------------------------------

        logging.info(
            "-" * 70
        )

        for client_id, result in client_results.items():

            logging.info(
                f"{client_id} | "
                f"AUC: {result['auc']:.4f}"
            )

        # --------------------------------------------------------
        # Mean client AUC
        # --------------------------------------------------------

        auc_values = [
            result["auc"]
            for result in client_results.values()
        ]

        if len(auc_values) > 0:

            mean_auc = float(
                np.mean(auc_values)
            )

            logging.info(
                f"Mean Client AUC: {mean_auc:.4f}"
            )

        else:

            mean_auc = 0.0

        logging.info(
            "=" * 70
        )

        return {
            "clients": client_results,
            "mean_auc": mean_auc
        }

    # ============================================================
    # HYBRID EVALUATION
    # ============================================================

    def _eval_hybrid(
        self,
        train_loader,
        test_loader
    ):

        if train_loader is None:

            raise ValueError(
                "'train_loader' is required "
                "for Hybrid evaluation."
            )

        train_latents = []
        test_latents = []
        test_labels = []

        self.model.eval()

        with torch.no_grad():

            # ----------------------------------------------------
            # Train latent space
            # ----------------------------------------------------

            for batch_input in tqdm(
                train_loader,
                desc="Extracting Train Latents..."
            ):

                batch_data = (
                    batch_input[0]
                    .to(self.device)
                )

                latent, _ = (
                    self._extract_features_and_output(
                        batch_data
                    )
                )

                train_latents.append(
                    latent
                )

            # ----------------------------------------------------
            # Test latent space
            # ----------------------------------------------------

            for batch_input in tqdm(
                test_loader,
                desc="Extracting Test Latents..."
            ):

                batch_data = (
                    batch_input[0]
                    .to(self.device)
                )

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

        cen = CentroidBasedOneClassClassifier()

        cen.fit(
            train_latents
        )

        start_time = time.time()

        predictions_cen = cen.get_density(
            test_latents
        )

        infer_time = (
            time.time()
            - start_time
        )

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

        self.model.eval()

        results = {}

        # --------------------------------------------------------
        # Autoencoder / VAE
        # --------------------------------------------------------

        if self.model_type in [
            "autoencoder",
            "both"
        ]:

            results["autoencoder"] = (
                self._eval_autoencoder(
                    test_loader
                )
            )

        # --------------------------------------------------------
        # Hybrid
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Comparison
        # --------------------------------------------------------

        self._log_comparison(
            results
        )

        # --------------------------------------------------------
        # Return
        # --------------------------------------------------------

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
