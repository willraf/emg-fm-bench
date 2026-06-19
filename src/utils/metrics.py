from sklearn import metrics as skmetrics
import torch
import torch.distributed as dist
import numpy as np


class Metrics:
    """
    Base metrics class. Handles accumulation of predictions and targets,
    and distributed gathering. Subclasses implement compute().
    """

    def __init__(self, device, distributed=False, is_main=False):
        self.distributed = distributed
        self.device = device
        self.is_main = is_main
        self.reset()

    def update(self, preds, targets, loss=None, num_samples=None):
        self.preds.append(preds.detach().cpu())
        self.targets.append(targets.detach().cpu())
        if loss is not None:
            samples = num_samples if num_samples else targets.size(0)
            self.total_loss += loss * samples
            self.total_samples += samples

    def _gather(self):
        """
        Gathers predictions, targets and loss across distributed processes.
        Returns (preds_np, targets_np, final_loss) on main process,
        returns None on non-main processes in distributed mode.
        """
        local_preds   = torch.cat(self.preds)
        local_targets = torch.cat(self.targets)

        if self.distributed:
            output_preds   = [None for _ in range(dist.get_world_size())]
            output_targets = [None for _ in range(dist.get_world_size())]

            dist.all_gather_object(output_preds,   local_preds)
            dist.all_gather_object(output_targets, local_targets)

            loss_tensor = torch.tensor(
                [self.total_loss, self.total_samples], device=self.device
            )
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

            if not self.is_main:
                return None, None, None

            preds   = torch.cat(output_preds).numpy()
            targets = torch.cat(output_targets).numpy()
            final_loss = (loss_tensor[0] / loss_tensor[1]).item() \
                         if loss_tensor[1] > 0 else 0.0
        else:
            preds      = local_preds.numpy()
            targets    = local_targets.numpy()
            final_loss = (self.total_loss / self.total_samples) \
                         if self.total_samples > 0 else 0.0

        if isinstance(final_loss, torch.Tensor):
            final_loss = final_loss.item()

        return preds, targets, final_loss

    def compute(self):
        raise NotImplementedError("Subclasses must implement compute()")

    def reset(self):
        self.preds         = []
        self.targets       = []
        self.total_loss    = 0.0
        self.total_samples = 0


# --------------------------------------------------------------------------- #


class ClassificationMetrics(Metrics):
    """
    Metrics for classification tasks.
    Expects:
        preds:   (B, K) raw logits
        targets: (B,)   class indices
    """

    def update(self, preds, targets, loss=None, num_samples=None):
        # Convert logits to class indices before accumulating
        class_preds = preds.argmax(dim=1)  # (B,)
        super().update(class_preds, targets, loss=loss, num_samples=num_samples)

    def compute(self):
        preds, targets, final_loss = self._gather()

        if preds is None:
            return None

        acc = skmetrics.accuracy_score(targets, preds)
        prec, recall, f1, _ = skmetrics.precision_recall_fscore_support(
            targets, preds, average='macro', zero_division=0
        )
        cm = skmetrics.confusion_matrix(targets, preds)

        return {
            "loss":             final_loss,
            "acc":              acc,
            "f1":               f1,
            "precision":        prec,
            "recall":           recall,
            "confusion_matrix": cm,
        }

# --------------------------------------------------------------------------- #

class RegressionMetrics(Metrics):
    """
    Metrics for per-patch regression tasks (pose / force).

    Expects:
        preds:   (B, N, J)  — per-patch predictions
        targets: (B, N, J)  — per-patch targets
            B = batch size, N = number of patches, J = output dims (joints/force channels)

    Reports:
        loss:                mean MSE loss
        rmse:                mean per-joint RMSE averaged across joints
        mae:                 mean per-joint MAE averaged across joints
        pearson_r:           mean per-joint Pearson r averaged across joints
        r2:                  mean per-joint R² averaged across joints
        rmse_per_joint:      (J,) array
        mae_per_joint:       (J,) array
        pearson_r_per_joint: (J,) array
        r2_per_joint:        (J,) array
    """

    def compute(self):
        preds, targets, final_loss = self._gather()

        if preds is None:
            return None

        total_windows, N, J = preds.shape

        # Flatten to (total_windows * N, J) for per-joint stats
        preds_flat   = preds.reshape(-1, J)    # (M, J)  M = total_windows * N
        targets_flat = targets.reshape(-1, J)  # (M, J)

        rmse_per_joint    = self._rmse(preds_flat, targets_flat)       # (J,)
        mae_per_joint     = self._mae(preds_flat, targets_flat)        # (J,)
        pearson_r_per_joint = self._pearson_r(preds_flat, targets_flat) # (J,)
        r2_per_joint      = self._r2(preds_flat, targets_flat)         # (J,)

        return {
            "loss":                  final_loss,
            "rmse":                  float(rmse_per_joint.mean()),
            "mae":                   float(mae_per_joint.mean()),
            "pearson_r":             float(pearson_r_per_joint.mean()),
            "r2":                    float(r2_per_joint.mean()),
            "rmse_per_joint":        rmse_per_joint,
            "mae_per_joint":         mae_per_joint,
            "pearson_r_per_joint":   pearson_r_per_joint,
            "r2_per_joint":          r2_per_joint,
        }

    # ---------------------------------------------------------------------- #
    # Static metric implementations
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _rmse(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Per-joint RMSE. Shape: (J,)"""
        return np.sqrt(np.mean((preds - targets) ** 2, axis=0))

    @staticmethod
    def _mae(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Per-joint MAE. Shape: (J,)"""
        return np.mean(np.abs(preds - targets), axis=0)

    @staticmethod
    def _pearson_r(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Per-joint Pearson r. Shape: (J,)
        Guards against zero-variance joints (e.g. a joint that never moves).
        """
        preds_z   = preds   - preds.mean(axis=0, keepdims=True)
        targets_z = targets - targets.mean(axis=0, keepdims=True)

        num   = (preds_z * targets_z).sum(axis=0)
        denom = (
            np.sqrt((preds_z ** 2).sum(axis=0)) *
            np.sqrt((targets_z ** 2).sum(axis=0))
        )
        r = np.where(denom > 1e-8, num / denom, 0.0)
        return np.clip(r, -1.0, 1.0)

    @staticmethod
    def _r2(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Per-joint coefficient of determination R². Shape: (J,)

        R² = 1 - SS_res / SS_tot
            SS_res = sum of squared residuals
            SS_tot = total sum of squares (variance of targets)

        R² = 1.0  → perfect prediction
        R² = 0.0  → model predicts the mean (no better than baseline)
        R² < 0.0  → model is worse than predicting the mean

        Guards against zero-variance targets (constant joint) by returning
        0.0 rather than NaN, consistent with sklearn convention.
        """
        ss_res = ((targets - preds) ** 2).sum(axis=0)           # (J,)
        ss_tot = ((targets - targets.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)  # (J,)
        r2 = np.where(ss_tot > 1e-8, 1.0 - ss_res / ss_tot, 0.0)
        return r2