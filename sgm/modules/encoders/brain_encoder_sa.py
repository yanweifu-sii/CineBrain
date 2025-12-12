import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """
        Pure self-attention
        x: (B, L, D)
        """
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn_logits = attn_logits.masked_fill(attn_mask == 0, float('-inf'))

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        if activation == 'gelu':
            self.act = F.gelu
        elif activation == 'relu':
            self.act = F.relu
        else:
            raise ValueError("Unsupported activation")

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerLayer(nn.Module):
    """
    Only self-attn + feedforward
    """
    def __init__(self, embed_dim: int, nhead: int, dim_feedforward: int, dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, nhead, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, dim_feedforward, dropout, activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, self_mask: Optional[torch.Tensor] = None):
        x_norm = self.norm1(x)
        sa = self.self_attn(x_norm, attn_mask=self_mask)
        x = x + self.dropout(sa)
        x_norm = self.norm2(x)
        ff = self.ff(x_norm)
        x = x + self.dropout(ff)
        return x


class fMRITokenizer(nn.Module):
    def __init__(self, in_channels: int = 5, seq_len: int = 8405,
                 embed_dim: int = 2048, num_spatial: int = 226):
        super().__init__()
        self.channel_proj = nn.Linear(in_channels, embed_dim)
        self.spatial_proj = nn.Linear(seq_len, num_spatial)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[1] != self.channel_proj.in_features:
            x = x.permute(0, 2, 1)  # (B, in_channels, seq_len)
        B = x.shape[0]
        x = x.permute(0, 2, 1)  # (B, seq_len, in_channels)
        x = self.channel_proj(x)  # (B, seq_len, D)
        x = x.permute(0, 2, 1)    # (B, D, seq_len)
        x = self.spatial_proj(x)  # (B, D, num_spatial)
        x = x.permute(0, 2, 1)    # (B, num_spatial, D)
        return x


class EEGTokenizer(nn.Module):
    def __init__(self, d_model: int = 2048, num_spatial: int = 226):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels=5*64, out_channels=d_model, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(num_spatial)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.view(B, 5*64, x.shape[-1])  # (B, 320, 800)
        x = self.input_proj(x)            # (B, D, 800)
        x = self.pool(x)                  # (B, D, num_spatial)
        x = x.permute(0, 2, 1)            # (B, num_spatial, D)
        return x


class BrainTransformer(nn.Module):
    """
    EEG + fMRI are concatenated, then pass through one shared Transformer.
    """
    def __init__(self,
                 in_fmri_channels: int = 5,
                 fmri_seq_len: int = 8405,
                 num_spatial: int = 226,
                 embed_dim: int = 4096,
                 out_dim: int = 2048,
                 clip_dim: int = 1152,
                 nhead: int = 32,
                 num_layers: int = 12,
                 dropout: float = 0.01):
        super().__init__()
        self.fmri_tokenizer = fMRITokenizer(in_channels=in_fmri_channels, seq_len=fmri_seq_len,
                                            embed_dim=embed_dim, num_spatial=num_spatial)
        self.eeg_tokenizer = EEGTokenizer(d_model=embed_dim, num_spatial=num_spatial)

        # CLS tokens
        self.cls_eeg = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.cls_fmri = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Position embeddings for combined sequence
        self.pos_emb = nn.Parameter(torch.randn(1, 2 * (num_spatial + 1), embed_dim))

        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, nhead, embed_dim * 4, dropout, activation='gelu')
            for _ in range(num_layers)
        ])

        self.eeg_out_proj = nn.Linear(embed_dim, out_dim)
        self.fmri_out_proj = nn.Linear(embed_dim, out_dim)
        self.eeg_cls_proj = nn.Linear(embed_dim, clip_dim)
        self.fmri_cls_proj = nn.Linear(embed_dim, clip_dim)
        self.dropout = nn.Dropout(dropout)

        self.num_spatial = num_spatial

    def forward(self, eeg_x: torch.Tensor, fmri_x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = eeg_x.shape[0]
        eeg_tokens = self.eeg_tokenizer(eeg_x)   # (B, num_spatial, D)
        fmri_tokens = self.fmri_tokenizer(fmri_x)  # (B, num_spatial, D)

        cls_eeg = self.cls_eeg.expand(B, -1, -1)
        cls_fmri = self.cls_fmri.expand(B, -1, -1)

        eeg = torch.cat([cls_eeg, eeg_tokens], dim=1)   # (B, num_spatial+1, D)
        fmri = torch.cat([cls_fmri, fmri_tokens], dim=1)

        # concat both modalities
        x = torch.cat([eeg, fmri], dim=1)  # (B, 2*(num_spatial+1), D)
        x = x + self.pos_emb

        for layer in self.layers:
            x = layer(x, self_mask=mask)

        # split back
        eeg_out, fmri_out = x[:, :self.num_spatial+1, :], x[:, self.num_spatial+1:, :]

        eeg_cls = eeg_out[:, 0, :]
        fmri_cls = fmri_out[:, 0, :]

        eeg_feats = self.eeg_out_proj(eeg_out[:, 1:, :])
        fmri_feats = self.fmri_out_proj(fmri_out[:, 1:, :])

        eeg_cls_proj = self.eeg_cls_proj(eeg_cls)
        fmri_cls_proj = self.fmri_cls_proj(fmri_cls)

        return eeg_cls_proj, fmri_cls_proj, eeg_feats, fmri_feats
