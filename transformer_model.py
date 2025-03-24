import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.embedding_short = nn.Linear(input_dim, model_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.transformer_encoder_short = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, output_dim)
        self.fc_short = nn.Linear(model_dim, output_dim)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_avg_pool_short = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, short_x):
        # x.shape = [B, seq_len, input_dim]
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.fc(x)  # [B, seq_len, output_dim]
        x = x.permute(0, 2, 1)  # 方便用 1d pooling,变成 [B, output_dim, seq_len]
        pooled_output = self.global_avg_pool(x)   # [B, output_dim, 1]
        pooled_output = pooled_output.permute(0, 2, 1)  # [B, 1, output_dim]

        short_x = self.embedding_short(short_x)
        short_x = self.transformer_encoder_short(short_x)
        short_x = self.fc_short(short_x)    # [B, short_seq_len, output_dim]
        short_x = short_x.permute(0, 2, 1)
        short_pooled_output = self.global_avg_pool_short(short_x)   # [B, output_dim, 1]
        short_pooled_output = short_pooled_output.permute(0, 2, 1)  # [B, 1, output_dim]

        # 最终把长期与短期的 pooled 结果加起来
        pooled_output = pooled_output + short_pooled_output   # [B, 1, output_dim]
        # 如果需要返回序列全部时刻的输出用于 topk，则可以保留 x
        # 这里我们返回 [B, seq_len, output_dim] 也行，或者只返回 pooled。
        # 为了与 Transformer 写法统一，这里返回 [B, seq_len, output_dim] 形状
        # 可以把 pooled_output broadcast 回 seq_len 大小以兼容你的 loss 计算逻辑。
        # 也可只返回 pooled_output 让外部做分类（看你的设计）。
        # 这里假设我们把 pooled_output repeat成 seq_len 时间步，方便外面取 [:, -1, :]

        seq_len = x.shape[-1]
        # pooled_output.shape = [B, 1, output_dim]
        pooled_output = pooled_output.repeat(1, seq_len, 1)  # [B, seq_len, output_dim]

        return pooled_output

