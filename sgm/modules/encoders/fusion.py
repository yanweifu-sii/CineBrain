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
    
class CrossModalFusionTransformer(nn.Module):
    """跨模态融合Transformer，用于融合fMRI和EEG特征"""
    def __init__(
        self,
        input_dim=4096,  # fMRI(2048) + EEG(2048) = 4096
        hidden_dim=2048,
        output_dim=4096,
        num_heads=16,
        num_layers=6,
        dropout=0.01
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.modality_embed = nn.Parameter(torch.randn(2, hidden_dim))  # fMRI和EEG的模态嵌入
        self.pos_embed = nn.Parameter(torch.randn(1, 452, hidden_dim))  # 226*2 = 452个位置
        
        # Fusion transformer layers
        self.fusion_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=4*hidden_dim,
                dropout=dropout,
                activation='gelu'
            ) for _ in range(num_layers)
        ])
        
        # 输出投射
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # # 池化层用于获得全局特征
        # self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, fmri_embedding, eeg_embedding):
        """
        Args:
            fmri_embedding: (B, 226, 2048) - fMRI spatial embeddings
            eeg_embedding: (B, 226, 2048) - EEG spatial embeddings
        Returns:
            fused_features: (B, 452, 2048) - 融合后的特征
            global_features: (B, 2048) - 全局融合特征
        """
        B = fmri_embedding.shape[0]
        
        # 连接fMRI和EEG特征
        combined_features = torch.cat([fmri_embedding, eeg_embedding], dim=-1)  # (B, 226, 4096)
        combined_features = self.input_proj(combined_features)  # (B, 226, 2048)
        
        # 为每个模态添加不同的模态嵌入
        fmri_features = combined_features + self.modality_embed[0]  # (B, 226, 2048)
        eeg_features = combined_features + self.modality_embed[1]   # (B, 226, 2048)
        
        # 拼接两个模态的特征
        fused_features = torch.cat([fmri_features, eeg_features], dim=1)  # (B, 452, 2048)
        
        # 添加位置编码
        fused_features = fused_features + self.pos_embed
        
        # 通过fusion transformer层
        for layer in self.fusion_layers:
            fused_features = layer(fused_features)

        fused_features = self.norm(fused_features)
        fused_features = self.output_proj(fused_features)  # (B, 452, 4096)
        
        # 计算全局特征
        # global_features = self.global_pool(fused_features.transpose(1, 2)).squeeze(-1)  # (B, 2048)
        
        return fused_features