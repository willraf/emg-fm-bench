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
            # 1. Gather all predictions and targets to rank 0
            # We use a list-based gather which is safer for varying sizes
            output_preds = [None for _ in range(dist.get_world_size())]
            output_targets = [None for _ in range(dist.get_world_size())]
            
            # all_gather_object is slower than all_gather but handles 
            # the "dangling batch" size difference automatically
            dist.all_gather_object(output_preds, local_preds)
            dist.all_gather_object(output_targets, local_targets)

            if not self.is_main:
                return None # Only main process computes final metrics

            preds = torch.cat(output_preds).numpy()
            targets = torch.cat(output_targets).numpy()
        else:    
            preds = local_preds.numpy()
            targets = local_targets.numpy()

        acc = metrics.accuracy_score(targets, preds)
        prec, recall, f1, _ = metrics.precision_recall_fscore_support(
            targets, preds, average='macro', zero_division=0
        )
        
        cm = metrics.confusion_matrix(targets, preds)

        if self.distributed:
            # You must also reduce the loss and sample counts across GPUs
            loss_tensor = torch.tensor([self.total_loss, self.total_samples], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            final_loss = (loss_tensor[0] / loss_tensor[1])
        else:
            final_loss = self.total_loss / self.total_samples if self.total_samples > 0 else 0
        
        final_loss = final_loss.item()

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