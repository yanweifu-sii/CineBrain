import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer

class EEGTransformer(nn.Module):
    def __init__(
            self, 
            d_model=2048, 
            out_dim=2048,
            clip_dim=1152,
            nhead=32, 
            num_layers=24, 
            dropout=0.01
        ):
        super().__init__()
        
        # 输入处理模块
        self.input_proj = nn.Conv1d(
            in_channels=5*64,       # 合并5个信号和64个通道
            out_channels=d_model,
            kernel_size=1
        )
        self.pool = nn.AdaptiveAvgPool1d(226) 
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4*d_model,
                dropout=dropout,
                batch_first=True
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
        
        eeg_cls = None
        for layer_idx, layer in enumerate(self.encoder_layers):
            x = layer(x)
            
            # 在第12层提取CLS Token特征
            if layer_idx == 11:  # 注意索引从0开始计数
                eeg_cls = x[:, 0, :]  # (B, 2048)
        x = self.out_proj(x)[:, 1:]
        eeg_cls = self.cls_proj(eeg_cls)
        return eeg_cls, x  # 返回最终输出和中间特征