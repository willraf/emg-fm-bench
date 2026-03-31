import torch
from torch import nn

class AblationFM(nn.Module):
    """
    A lightweight replacement for NeuroRVQ to be used in BioFMWrapper.
    It simulates the tokenization process using either a Linear projection 
    or a small Depthwise Convolution.
    """
    def __init__(self, cfg):
        super().__init__()
        self.c_in = cfg.num_channels        # e.g., 8
        self.patch_size = cfg.patch_size    # e.g., 5
        self.embed_dim = cfg.embed_dim      # e.g., 200
        self.use_conv = cfg.get("use_conv_ablation", False)

        if self.use_conv:
            # OPTION 1: Small Conv followed by Linear
            # We use a Depthwise Conv to keep channels independent at first
            self.projector = nn.Sequential(
                nn.Conv1d(
                    in_channels=self.c_in, 
                    out_channels=self.c_in, 
                    kernel_size=3, 
                    padding=1, 
                    groups=self.c_in
                ),
                nn.ReLU(),
                nn.Flatten(0, 1), # Flatten B and C for the Linear layer
                nn.Linear(self.patch_size, self.embed_dim)
            )
        else:
            # OPTION 2: Pure Linear Head (Raw Patch Projection)
            self.projector = nn.Linear(self.patch_size, self.embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, T) where T is total time steps (e.g. 40)
        Returns:
            tokens: (B, N, C, D) 
            Note: Wrapper expects (B, N, C, D) based on your tokens.shape line
        """
        B, C, T = x.shape
        N = T // self.patch_size # Number of patches (e.g., 40 / 5 = 8)

        # 1. Reshape into patches: (B, C, N, patch_size)
        x = x.view(B, C, N, self.patch_size)

        if self.use_conv:
            # Prepare for Conv: (B*C, 1, T)
            # This is a bit more complex because convs expect (B, C, T)
            # We treat each channel-patch as a mini-sequence
            x = x.reshape(B * C, 1, N * self.patch_size) 
            # ... apply projector logic ...
            # For a pure ablation, let's keep it simple:
            tokens = self.projector(x.view(B*C, N, self.patch_size))
        else:
            # Linear projection of the patch dimension
            # (B, C, N, patch_size) -> (B, C, N, embed_dim)
            tokens = self.projector(x)

        # Final reshape to match BioFMWrapper expectation: (B, N, C, D)
        # Your wrapper uses B, C, N, D = tokens.shape, then tokens.mean(dim=(1,2))
        # This matches: (Batch, Num_Patches, Channels, Embed_Dim)
        return tokens.permute(0, 2, 1, 3) 

    def load_checkpoint(self, path):
        # Placeholder to keep the Trainer happy
        logging.info("Ablation model initialized from scratch.")
        pass