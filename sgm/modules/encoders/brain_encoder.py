import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention that supports both self-attention and cross-attention.
    forward(query, key_value=None, attn_mask=None)
    - if key_value is None -> self-attention (K=V=Q)
    - else -> cross-attention (Q=query, K=key_value, V=key_value)
    """
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

    def forward(self, query: torch.Tensor, key_value: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        """
        query: (B, Lq, D)
        key_value: None or (B, Lk, D)
        attn_mask: (B, num_heads, Lq, Lk) or broadcastable mask where 0 means masked
        """
        if key_value is None:
            key_value = query

        B, Lq, D = query.shape
        Lk = key_value.shape[1]

        q = self.q_proj(query).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, Lq, hd)
        k = self.k_proj(key_value).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, Lk, hd)
        v = self.v_proj(key_value).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, Lk, hd)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, nh, Lq, Lk)

        if attn_mask is not None:
            # assume mask==0 means masked
            attn_logits = attn_logits.masked_fill(attn_mask == 0, float('-inf'))

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, nh, Lq, hd)
        out = out.transpose(1, 2).contiguous().view(B, Lq, D)  # (B, Lq, D)
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


class TransformerLayerWithCross(nn.Module):
    """
    Single transformer layer for one modality with optional cross-attention to the other modality.
    Operation (pre-norm):
      - self-attn -> add
      - cross-attn (using other modality) -> add
      - feed-forward -> add
    """
    def __init__(self, embed_dim: int, nhead: int, dim_feedforward: int, dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, nhead, dropout)
        self.cross_attn = MultiHeadAttention(embed_dim, nhead, dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.ff = FeedForward(embed_dim, dim_feedforward, dropout, activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, other: Optional[torch.Tensor] = None, self_mask: Optional[torch.Tensor] = None, cross_mask: Optional[torch.Tensor] = None):
        """
        x: (B, Lx, D)
        other: (B, Ly, D) or None
        """
        # self-attn (pre-norm)
        x_norm = self.norm1(x)
        sa = self.self_attn(x_norm, None, attn_mask=self_mask)
        x = x + self.dropout(sa)

        # cross-attn if other provided
        if other is not None:
            x_norm = self.norm2(x)
            ca = self.cross_attn(x_norm, other, attn_mask=cross_mask)  # Q=x, K=V=other
            x = x + self.dropout(ca)

        # feed-forward
        x_norm = self.norm3(x)
        ff = self.ff(x_norm)
        x = x + self.dropout(ff)
        return x


# Tokenizers (as you provided, wrapped slightly)
class fMRITokenizer(nn.Module):
    def __init__(self,
                 in_channels: int = 5,
                 seq_len: int = 8405,
                 embed_dim: int = 2048,
                 num_spatial: int = 226):
        super().__init__()
        self.channel_proj = nn.Linear(in_channels, embed_dim)
        self.spatial_proj = nn.Linear(seq_len, num_spatial)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # expects x shape (B, in_channels, seq_len) or (B, seq_len, in_channels)
        if x.dim() == 3 and x.shape[1] != self.channel_proj.in_features:
            # If input is (B, seq_len, in_channels)
            x = x.permute(0, 2, 1)  # -> (B, in_channels, seq_len)
        # Now x is (B, in_channels, seq_len)
        B = x.shape[0]
        x = x.permute(0, 2, 1)  # (B, seq_len, in_channels)
        x = self.channel_proj(x)  # (B, seq_len, embed_dim)
        x = x.permute(0, 2, 1)    # (B, embed_dim, seq_len)
        x = self.spatial_proj(x)  # (B, embed_dim, num_spatial)
        x = x.permute(0, 2, 1)    # (B, num_spatial, embed_dim)
        return x


class EEGTokenizer(nn.Module):
    def __init__(self, d_model: int = 2048, num_spatial: int = 226):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels=5*64, out_channels=d_model, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(num_spatial)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # expects x shape (B, 5, 64, 800)
        B = x.shape[0]
        x = x.view(B, 5*64, x.shape[-1])  # (B, 320, 800)
        x = self.input_proj(x)            # (B, d_model, 800)
        x = self.pool(x)                  # (B, d_model, num_spatial)
        x = x.permute(0, 2, 1)            # (B, num_spatial, d_model)
        return x


class BrainTransformer(nn.Module):
    """
    Integrates EEG and fMRI branches with cross-attention between them.
    Returns:
      eeg_cls (B, clip_dim), fmri_cls (B, clip_dim), eeg_tokens (B, num_spatial, out_dim), fmri_tokens (B, num_spatial, out_dim)
    """
    def __init__(
        self,
        in_fmri_channels: int = 5,
        fmri_seq_len: int = 8405,
        num_spatial: int = 226,
        embed_dim: int = 4096,
        out_dim: int = 2048,
        clip_dim: int = 1152,
        nhead: int = 32,
        num_layers: int = 12,
        dropout: float = 0.01
    ):
        super().__init__()

        # tokenizers
        self.fmri_tokenizer = fMRITokenizer(in_channels=in_fmri_channels, seq_len=fmri_seq_len, embed_dim=embed_dim, num_spatial=num_spatial)
        self.eeg_tokenizer = EEGTokenizer(d_model=embed_dim, num_spatial=num_spatial)

        # cls tokens and position embeddings for both modalities
        self.cls_eeg = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.cls_fmri = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_eeg = nn.Parameter(torch.randn(1, num_spatial + 1, embed_dim))
        self.pos_fmri = nn.Parameter(torch.randn(1, num_spatial + 1, embed_dim))

        # stacks of layers (shared structure for both branches)
        self.eeg_layers = nn.ModuleList([
            TransformerLayerWithCross(embed_dim, nhead, embed_dim * 4, dropout, activation='gelu')
            for _ in range(num_layers)
        ])
        self.fmri_layers = nn.ModuleList([
            TransformerLayerWithCross(embed_dim, nhead, embed_dim * 4, dropout, activation='gelu')
            for _ in range(num_layers)
        ])

        # output projections
        self.eeg_out_proj = nn.Linear(embed_dim, out_dim)
        self.fmri_out_proj = nn.Linear(embed_dim, out_dim)

        # cls projection to clip dim (for contrastive / downstream)
        self.eeg_cls_proj = nn.Linear(embed_dim, clip_dim)
        self.fmri_cls_proj = nn.Linear(embed_dim, clip_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, eeg_x: torch.Tensor, fmri_x: torch.Tensor,
                eeg_mask: Optional[torch.Tensor] = None, fmri_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        eeg_x: (B, 5, 64, 800)
        fmri_x: (B, seq_len, in_channels) OR (B, in_channels, seq_len)
        masks optional (not implemented in detail here) - could be used for padding
        """

        B = eeg_x.shape[0]

        # Tokenize
        eeg_tokens = self.eeg_tokenizer(eeg_x)   # (B, num_spatial, D)
        fmri_tokens = self.fmri_tokenizer(fmri_x)  # (B, num_spatial, D)

        # prepend cls tokens
        cls_eeg = self.cls_eeg.expand(B, -1, -1)   # (B,1,D)
        cls_fmri = self.cls_fmri.expand(B, -1, -1)
        eeg = torch.cat([cls_eeg, eeg_tokens], dim=1)   # (B, num_spatial+1, D)
        fmri = torch.cat([cls_fmri, fmri_tokens], dim=1)

        # add positional encodings
        eeg = eeg + self.pos_eeg
        fmri = fmri + self.pos_fmri

        # iterate layers: each layer does self-attn + cross-attn (both directions)
        for idx, (elayer, player) in enumerate(zip(self.eeg_layers, self.fmri_layers)):
            # Note: we let cross-attn use the most recent representation from the other branch
            eeg = elayer(eeg, other=fmri, self_mask=eeg_mask, cross_mask=None)
            fmri = player(fmri, other=eeg, self_mask=fmri_mask, cross_mask=None)

        # extract cls tokens
        eeg_cls = eeg[:, 0, :]  # (B, D)
        fmri_cls = fmri[:, 0, :]

        # project outputs
        eeg_feats = self.eeg_out_proj(eeg[:, 1:, :])   # (B, num_spatial, out_dim)
        fmri_feats = self.fmri_out_proj(fmri[:, 1:, :])  # (B, num_spatial, out_dim)

        eeg_cls_proj = self.eeg_cls_proj(eeg_cls)   # (B, clip_dim)
        fmri_cls_proj = self.fmri_cls_proj(fmri_cls)  # (B, clip_dim)

        return eeg_cls_proj, fmri_cls_proj, eeg_feats, fmri_feats

