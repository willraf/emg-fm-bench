from sklearn import metrics
import torch
import torch.distributed as dist

class Metrics:
    """
    Handles metric tracking through
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
            # We store the sum of losses to get a true mathematical mean later
            # loss is usually batch-averaged, so multiply by batch size
            samples = num_samples if num_samples else targets.size(0)
            self.total_loss += loss * samples
            self.total_samples += samples


    def compute(self):
        local_preds = torch.cat(self.preds)
        local_targets = torch.cat(self.targets)
        
        if self.distributed:
            # Gather all predictions and targets to rank 0
            # We use a list-based gather which is safer for varying sizes
            output_preds = [None for _ in range(dist.get_world_size())]
            output_targets = [None for _ in range(dist.get_world_size())]
            
            # all_gather_object is slower than all_gather but handles 
            # the "dangling batch" size difference automatically
            dist.all_gather_object(output_preds, local_preds)
            dist.all_gather_object(output_targets, local_targets)

            # must also reduce the loss and sample counts across GPUs
            loss_tensor = torch.tensor([self.total_loss, self.total_samples], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

            if not self.is_main:
                return None # Only main process computes final metrics

            preds = torch.cat(output_preds).numpy()
            targets = torch.cat(output_targets).numpy()
            final_loss = (loss_tensor[0] / loss_tensor[1]).item() if loss_tensor[1] > 0 else 0
        else:    
            preds = local_preds.numpy()
            targets = local_targets.numpy()
            final_loss = self.total_loss / self.total_samples if self.total_samples > 0 else 0

        final_loss = final_loss.item() if isinstance(final_loss, torch.Tensor) else final_loss

        acc = metrics.accuracy_score(targets, preds)
        prec, recall, f1, _ = metrics.precision_recall_fscore_support(
            targets, preds, average='macro', zero_division=0
        )
        
        cm = metrics.confusion_matrix(targets, preds)


        return {
            "loss": final_loss,
            "acc": acc,
            "f1": f1,
            "precision": prec,
            "recall": recall,
            "confusion_matrix": cm,
            # Only return raw targets and preds if we really need them e.g. for confusion matrix
            # "targets": targets,
            # "preds": preds,
        }
    
    def reset(self):
        self.preds = []
        self.targets = []
        self.total_loss = 0.0
        self.total_samples = 0



class RegressionMetrics:
    """
    Tracks MSE loss, per-joint RMSE and mean Pearson r across batches.
    Accumulates predictions and targets, computes metrics at epoch end.
    """
    def __init__(self, device, distributed=False, is_main=False):
        self.device = device
        self.distributed = distributed
        self.is_main = is_main
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.total_samples = 0
        # Accumulate for Pearson r computation
        self._all_preds = []   # list of (B, N, J) tensors
        self._all_targets = [] # list of (B, N, J) tensors

    def update(self, preds: torch.Tensor, targets: torch.Tensor,
               loss: torch.Tensor, num_samples: int):
        """
        Args:
            preds:   (B, N, J)
            targets: (B, N, J)
            loss:    scalar tensor
            num_samples: B
        """
        self.total_loss += loss.item() * num_samples
        self.total_samples += num_samples
        # Detach and move to CPU for accumulation
        self._all_preds.append(preds.detach().cpu())
        self._all_targets.append(targets.detach().cpu())

    def compute(self) -> dict:
        """
        Returns dict with:
            mean_loss:  average MSE loss
            rmse:       per-joint RMSE averaged across joints  (scalar)
            pearson_r:  per-joint Pearson r averaged across joints (scalar)
        """
        if self.total_samples == 0:
            return {"mean_loss": 0.0, "rmse": 0.0, "pearson_r": 0.0}

        # Stack: (total_windows, N, J)
        all_preds   = torch.cat(self._all_preds,   dim=0)
        all_targets = torch.cat(self._all_targets, dim=0)

        # Flatten to (total_windows * N, J) for per-joint stats
        B_N = all_preds.shape[0] * all_preds.shape[1]
        J   = all_preds.shape[2]
        preds_flat   = all_preds.reshape(B_N, J)
        targets_flat = all_targets.reshape(B_N, J)

        # Per-joint RMSE
        rmse_per_joint = torch.sqrt(
            ((preds_flat - targets_flat) ** 2).mean(dim=0)
        )  # (J,)
        mean_rmse = rmse_per_joint.mean().item()

        # Per-joint Pearson r
        pearson_r_per_joint = self._pearson_r(preds_flat, targets_flat)  # (J,)
        mean_pearson_r = pearson_r_per_joint.mean().item()

        mean_loss = self.total_loss / self.total_samples

        return {
            "mean_loss":    mean_loss,
            "rmse":         mean_rmse,
            "pearson_r":    mean_pearson_r,
            "rmse_per_joint":     rmse_per_joint.numpy(),
            "pearson_r_per_joint": pearson_r_per_joint.numpy(),
        }

    @staticmethod
    def _pearson_r(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute per-column (per-joint) Pearson correlation.

        Args:
            preds:   (N, J)
            targets: (N, J)
        Returns:
            r: (J,)
        """
        preds_z   = preds   - preds.mean(dim=0, keepdim=True)
        targets_z = targets - targets.mean(dim=0, keepdim=True)

        num   = (preds_z * targets_z).sum(dim=0)
        denom = (
            torch.sqrt((preds_z ** 2).sum(dim=0)) *
            torch.sqrt((targets_z ** 2).sum(dim=0))
        )
        # Clamp denominator to avoid division by zero on zero-variance joints
        r = num / denom.clamp(min=1e-8)
        return r.clamp(-1.0, 1.0)
