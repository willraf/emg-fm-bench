from torch import nn
from src.models.neuro_rvq_fm import NeuroRVQFM
import torch 

class NeuroRVQClassifierWrapper(nn.Module):
    def __init__(self, cfg, fm):
        super().__init__()
        self.fm = fm

        
        self.embed_dim = cfg.model.fm.embed_dim
        self.num_classes = cfg.finetuning.n_classes
        self.n_patches = cfg.model.fm.n_patches
        self.pooling_type = cfg.finetuning.pooling_type

        self.embed_dim = self.embed_dim * 4 if self.fm.use_as_encoder else self.embed_dim
        
        # We need to know how many channels are in your EPN data
        # Based on your previous context, this is 8
        self.n_channels = 8 
        self.n_time_steps = self.n_patches // self.n_channels

        # Head Definition
        if self.pooling_type == "tiny_myo_concat":
            # Concat across channels: dim = embed_dim * n_channels
            self.fc_norm = nn.LayerNorm(self.embed_dim * self.n_channels)
            self.head = nn.Linear(self.embed_dim * self.n_channels, self.num_classes)
        elif self.pooling_type == "attention_pool":
            self.query = nn.Parameter(torch.randn(1, 1, self.embed_dim))
            self.attn = nn.MultiheadAttention(self.embed_dim, num_heads=8, batch_first=True)
            self.fc_norm = nn.LayerNorm(self.embed_dim)
            self.head = nn.Linear(self.embed_dim, self.num_classes)
        else: # "mean" or "tiny_myo_mean"
            self.fc_norm = nn.LayerNorm(self.embed_dim)
            self.head = nn.Linear(self.embed_dim, self.num_classes)

    def forward(self, x, temporal_embedding_ix, spatial_embedding_ix):
        # 1. Get raw tokens from FM (Batch, 256, Embed_Dim)
        # We set return_patch_tokens=True to avoid the hardcoded mean(1) in their forward
        # Using branch_idx=0 as EPN uses the first branch logic
        tokens, _ = self.fm(x, temporal_embedding_ix, spatial_embedding_ix, 
                            return_patch_embeddings=True, branch_idx=0)
        
        # tokens shape: (B, 256, 200)
        B, N, E = tokens.shape
        
        # Reshape to (Batch, Channels, Time, Embed) -> (B, 8, 32, 200)
        tokens = tokens.reshape(B, self.n_channels, self.n_time_steps, E)

        if self.pooling_type == "mean":
            # Global Average across all patches (Simple & robust)
            feat = tokens.mean(dim=(1, 2)) # (B, 200)
            
        elif self.pooling_type == "tiny_myo_mean":
            # Mean across channels, then mean across time
            feat = tokens.mean(dim=1).mean(dim=1) # (B, 200)
            
        elif self.pooling_type == "tiny_myo_concat":
            # Concat channels: (B, Time, Chan*Embed) -> (B, 32, 1600)
            feat = tokens.permute(0, 2, 1, 3).reshape(B, self.n_time_steps, -1)
            feat = feat.mean(dim=1) # Average over time -> (B, 1600)

        elif self.pooling_type == "attention_pool":
            # Strategy 4: Model learns which patches represent the gesture
            # Flatten to (B, 256, 200) for attention
            flat_tokens = tokens.reshape(B, N, E)
            q = self.query.expand(B, -1, -1)
            feat, _ = self.attn(q, flat_tokens, flat_tokens)
            feat = feat.squeeze(1) # (B, 200)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        return self.head(self.fc_norm(feat))
    

    def get_fine_tuning_parameters(self):
        
        head_params = list(self.head.parameters()) + list(self.fc_norm.parameters())
        if self.pooling_type == "attention_pool":
            head_params += [self.query] + list(self.attn.parameters())

        return head_params