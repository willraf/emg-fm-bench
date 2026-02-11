from torch.nn.parallel import DistributedDataParallel as DDP 

class BaseTask:
    def train_step(self, model, batch, device):
        raise NotImplementedError
    
    def val_step(self, model, batch, device):
        raise NotImplementedError
    
    def eval_step(self, model, batch, device):
        raise NotImplementedError
    
    def wrap_distributed(self, local_rank):
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path):
        raise NotImplementedError

    def save_checkpoint(self, checkpoint_path, epoch, **kwargs):
        raise NotImplementedError
    

    def train(self):
        for m in self.modules():
            m.train()

    def eval(self):
        for m in self.modules():
            m.eval()

    def modules(self):
        return []
    
    @staticmethod
    def unwrap_model(model):
        """Utility function to unwrap a model from DDP if necessary.
        Prevents problems if loading a checkpoint saved from a DDP-wrapped model into a non-DDP model, or vice versa.
        """
        return model.module if isinstance(model, DDP) else model
    
    # TODO
    @property
    def monitor_metric(self):
        return "loss"