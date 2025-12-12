import torch
import torch.nn as nn

class TransformerBranch(nn.Module):
    def __init__(self, embed_dim=1920, num_layers=12, nhead=32):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=embed_dim*4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        cls_token = None
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == 11:  # 第11层输出分支
                cls_token = x[:, 0, :]
        return x, cls_token

class fMRITransformer(nn.Module):
    def __init__(self, 
        in_channels=5,
        seq_len=9447,
        out_dim=2048,
        embed_dim=2048,
        clip_dim=1152,
        num_spatial=226,
        num_layers=24
    ):
        super().__init__()
        # 通道和空间投影
        self.channel_proj = nn.Linear(in_channels, embed_dim)
        self.spatial_proj = nn.Linear(seq_len, num_spatial)
        
        # 类别token和位置编码
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_spatial+1, embed_dim))
        
        # Transformer编码器
        self.transformer = TransformerBranch(embed_dim, num_layers)
        self.out_proj = nn.Linear(embed_dim, out_dim)
        self.cls_proj = nn.Linear(embed_dim, clip_dim)
        
    def forward(self, x):
        B = x.shape[0]
        
        x = x.permute(0, 2, 1)  # (B, 9447, 5)
        x = self.channel_proj(x)  # (B, 9447, 1920)
        
        x = x.permute(0, 2, 1)   # (B, 1920, 9447)
        x = self.spatial_proj(x) # (B, 1920, 226)
        x = x.permute(0, 2, 1)   # (B, 226, 1920)
        
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, 1920)
        x = torch.cat([cls_token, x], dim=1)  # (B, 227, 1920)
        
        x += self.pos_embed
        
        x, mid_cls = self.transformer(x)
        mid_cls = self.cls_proj(mid_cls)
        x = self.out_proj(x)
        
        return mid_cls, x[:, 1:, :]  # (B, 226, 1920)