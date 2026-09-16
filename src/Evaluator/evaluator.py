import logging
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (
    roc_curve,
    auc,
    f1_score,
    precision_score,
    recall_score
)


class Evaluator(object):

    def __init__(
        self,
        model,
        model_type="both",
        metric="AUC",
        device=None
    ):
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
    # EXTRACT MODEL OUTPUT
    # ============================================================

    def _extract_features_and_output(self, batch_data):

        model_out = self.model(batch_data)

        # --------------------------------------------------------
        # VAE / Autoencoder
        #
        # Your VAE returns:
        #
        #     (output, mu, logvar)
        #
        # Therefore:
        #
        #     model_out[0] = reconstruction
        #     model_out[1] = mu
        #     model_out[2] = logvar
        # --------------------------------------------------------

        if isinstance(model_out, (tuple, list)):

            # Reconstruction
            output = model_out[0]

            # Latent representation
            #
            # For the VAE, mu is the deterministic latent
            # representation and is therefore used as latent
            # features when needed.
            if len(model_out) > 1:
                latent = model_out[1]
            else:
                latent = output

        else:

            output = model_out

            if hasattr(self.model, "encode"):

                encoded = self.model.encode(batch_data)

                # VAE encode() returns:
                #
                #     (mu, logvar)
                #
                # We use mu as the latent representation.
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

    def calculate_auc(
        self,
        y_true,
        score
    ):

        score = np.asarray(score)

        # --------------------------------------------------------
        # Safety check
        # --------------------------------------------------------

        if not np.all(np.isfinite(score)):

            logging.warning(
                "Non-finite anomaly scores detected. "
                "Replacing NaN/Inf values."
            )

            score = np.nan_to_num(
                score,
                nan=0.0,
                posinf=np.finfo(np.float64).max,
                neginf=np.finfo(np.float64).min
            )

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
            np.asarray(score) > threshold,
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

    def _eval_autoencoder(
        self,
        test_loader
    ):

        anomaly_scores = []
        test_labels = []

        self.model.eval()

        with torch.no_grad():

            for batch_input in tqdm(
                test_loader,
                desc="Evaluating Autoencoder..."
            ):

                # ------------------------------------------------
                # Input data
                # ------------------------------------------------

                batch_data = (
                    batch_input[0]
                    .to(self.device)
                )

                # ------------------------------------------------
                # Model forward pass
                #
                # VAE returns:
                #
                #     reconstruction, mu, logvar
                # ------------------------------------------------

                latent, output = (
                    self._extract_features_and_output(
                        batch_data
                    )
                )

                # ------------------------------------------------
                # Reconstruction error
                #
                # IMPORTANT:
                #
                # output = reconstructed X
                #
                # Therefore anomaly score is:
                #
                #     MSE(X, X_hat)
                #
                # NOT:
                #
                #     MSE(X, mu)
                # ------------------------------------------------

                reconstruction_error = torch.mean(
                    torch.nn.MSELoss(
                        reduction="none"
                    )(
                        batch_data,
                        output
                    ),
                    dim=1
                )

                anomaly_scores.append(
                    reconstruction_error
                )

                test_labels.append(
                    batch_input[1]
                )

        # ========================================================
        # Combine batches
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
        # AUC
        # ========================================================

        auc_val = self.calculate_auc(
            test_labels,
            anomaly_scores
        )

        # ========================================================
        # F1 / Precision / Recall
        #
        # Fixed threshold = 0.5
        # ========================================================

        (
            f1_val,
            prec_val,
            rec_val
        ) = self.calculate_classification_metrics(
            test_labels,
            anomaly_scores,
            threshold=0.5
        )

        # ========================================================
        # Return evaluation results
        # ========================================================

        return {
            "scores": anomaly_scores,
            "labels": test_labels,
            "auc": auc_val,
            "f1": f1_val,
            "precision": prec_val,
            "recall": rec_val
        }

    # ============================================================
    # PUBLIC EVALUATION FUNCTION
    # ============================================================

    def evaluate(
        self,
        test_loader
    ):

        if self.model_type in [
            "autoencoder",
            "vae",
            "ae"
        ]:

            return self._eval_autoencoder(
                test_loader
            )

        elif self.model_type == "hybrid":

            return self._eval_autoencoder(
                test_loader
            )

        elif self.model_type == "both":

            return self._eval_autoencoder(
                test_loader
            )

        else:

            raise ValueError(
                f"Unsupported model_type: "
                f"{self.model_type}"
            )
