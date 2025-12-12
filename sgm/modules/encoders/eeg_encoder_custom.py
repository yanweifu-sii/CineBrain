import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer


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
        # Self-attention block with pre-norm
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src_mask)
        src = src + self.dropout1(src2)
        
        # Feed-forward block with pre-norm
        src2 = self.norm2(src)
        src2 = self.feed_forward(src2)
        src = src + self.dropout2(src2)
        
        return src


class CustomEEGTransformer(nn.Module):
    def __init__(
        self, 
        d_model=2048,
        out_dim=2048,
        clip_dim=1152,
        nhead=32, 
        num_layers=12,  # 改为12层
        dropout=0.01,
        num_spatial=226
    ):
        super().__init__()
        
        # 输入处理模块
        self.input_proj = nn.Conv1d(
            in_channels=5*64,       # 合并5个信号和64个通道
            out_channels=d_model,
            kernel_size=1
        )
        self.pool = nn.AdaptiveAvgPool1d(num_spatial)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, num_spatial+1, d_model))  # 添加位置编码
        
        # 使用自定义Transformer层，与fMRI保持一致
        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4*d_model,
                dropout=dropout,
                activation='gelu'
            ) for _ in range(num_layers)
        ])
        
        self.out_proj = nn.Linear(d_model, out_dim)
        self.cls_proj = nn.Linear(d_model, clip_dim)
    
    def forward(self, x):
        # 输入形状: (B, 5, 64, 800)
        B = x.shape[0]
        x = x.view(B, 5*64, 800)    # (B, 320, 800)
        x = self.input_proj(x)      # (B, 2048, 800)
        x = self.pool(x)            # (B, 2048, 226)
        x = x.permute(0, 2, 1)      # (B, 226, 2048)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, 2048)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 227, 2048)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        eeg_cls = None
        for layer_idx, layer in enumerate(self.encoder_layers):
            x = layer(x)
            
            # 在第11层提取CLS Token特征（因为现在只有12层）
            if layer_idx == 11:  # 最后一层
                eeg_cls = x[:, 0, :]  # (B, 2048)
        
        x = self.out_proj(x)[:, 1:]  # (B, 226, 2048)
        eeg_cls = self.cls_proj(eeg_cls)  # (B, 1152)
        
        return eeg_cls, x