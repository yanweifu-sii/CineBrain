import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
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
    
    def forward(self, x, attn_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections and reshape for multi-head attention
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back and project to output dimension
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        
        return output


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1, activation='gelu'):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation='gelu'):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout, activation)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None):
        # Self-attention block
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src_mask)
        src = src + self.dropout1(src2)
        
        # Feed-forward block
        src2 = self.norm2(src)
        src2 = self.feed_forward(src2)
        src = src + self.dropout2(src2)
        
        return src


class CustomTransformerBranch(nn.Module):
    def __init__(self, embed_dim=1920, num_layers=12, nhead=32):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim*4,
                dropout=0.1,
                activation='gelu'
            ) for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        cls_token = None
        for i, layer in enumerate(self.layers):
            x = layer(x, mask)
            if i == 11:  # 11th layer (index 11) output branch
                cls_token = x[:, 0, :]
        return x, cls_token


class CustomfMRITransformer(nn.Module):
    def __init__(self,
                in_channels=5,
                seq_len=8405,
                out_dim=2048,
                embed_dim=2048,
                clip_dim=1152,
                num_spatial=226,
                num_layers=24,
                nhead=32
                ):
        super().__init__()
        # Channel and spatial projections
        self.channel_proj = nn.Linear(in_channels, embed_dim)
        self.spatial_proj = nn.Linear(seq_len, num_spatial)
        
        # Class token and positional encoding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_spatial+1, embed_dim))
        
        # Transformer encoder
        self.transformer = CustomTransformerBranch(embed_dim, num_layers, nhead)  # Using half the layers per branch
        self.out_proj = nn.Linear(embed_dim, out_dim)
        self.cls_proj = nn.Linear(embed_dim, clip_dim)
    
    def forward(self, x):
        B = x.shape[0]
        
        x = x.permute(0, 2, 1)  # (B, 9447, 5)
        x = self.channel_proj(x)  # (B, 9447, embed_dim)
        
        x = x.permute(0, 2, 1)   # (B, embed_dim, 9447)
        x = self.spatial_proj(x) # (B, embed_dim, 226)
        x = x.permute(0, 2, 1)   # (B, 226, embed_dim)
        
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_token, x], dim=1)  # (B, 227, embed_dim)
        
        x = x + self.pos_embed
        
        x, mid_cls = self.transformer(x)
        mid_cls = self.cls_proj(mid_cls)
        x = self.out_proj(x)
        
        return mid_cls, x[:, 1:, :]  # (B, 226, out_dim)
    
