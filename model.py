import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate=0.2):
        super().__init__()

        self.linear1 = nn.Linear(hidden_units, hidden_units)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.linear2 = nn.Linear(hidden_units, hidden_units)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x

        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.dropout2(x)

        x = residual + x

        return x


class SASRec(nn.Module):
    def __init__(self, itemnum, hidden_units, maxlen,
                 num_heads=2, num_blocks=2, dropout_rate=0.2):
        super().__init__()

        # 商品 Embedding
        self.item_embedding = nn.Embedding(
            num_embeddings=itemnum+1,
            embedding_dim=hidden_units,
            padding_idx=0
        )

        # 位置 Embedding
        self.position_embedding = nn.Embedding(
            num_embeddings=maxlen,
            embedding_dim=hidden_units
        )

        # Embedding 后的 Dropout
        self.dropout = nn.Dropout(p=dropout_rate)

        # 保存注意力头数，用于创建 Mask
        self.num_heads = num_heads

        # 多个 Transformer Block
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                # Attention 前的 LayerNorm
                "attention_norm": nn.LayerNorm(hidden_units),

                # 多头注意力层
                "attention": nn.MultiheadAttention(
                    embed_dim=hidden_units,
                    num_heads=num_heads,
                    dropout=dropout_rate,
                    batch_first=True
                ),

                # FFN 前的 LayerNorm
                "ffn_norm": nn.LayerNorm(hidden_units),

                # FFN
                "ffn": FeedForward(
                    hidden_units=hidden_units,
                    dropout_rate=dropout_rate
                )
            })
            for _ in range(num_blocks)
        ])

        # 所有 Block 之后的最终 LayerNorm
        self.final_layer_norm = nn.LayerNorm(hidden_units)

    def create_attention_mask(self, seq):
        batch_size, seq_len = seq.shape

        # 因果掩码，禁止关注未来位置
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=seq.device, dtype=torch.bool),
            diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).expand(
            batch_size, -1, -1
        ).clone()

        # 填充掩码，标记填充位置
        padding_mask = (seq == 0).unsqueeze(1)

        # 合并两个 mask
        mask = causal_mask | padding_mask

        # 防止后面 softmax出现全 True 行
        batch_idx, pos_idx = torch.where(seq == 0)
        mask[batch_idx, pos_idx, pos_idx] = False

        # 为每个注意力头复制mask
        mask = mask.repeat_interleave(
            self.num_heads,
            dim=0
        )

        return mask


    def forward(self, seq):
        # 商品 Embedding
        item_emb = self.item_embedding(seq) * (self.item_embedding.embedding_dim ** 0.5)

        # 位置 Embedding
        seq_len = seq.shape[1]
        positions = torch.arange(seq_len, device=seq.device)
        pos_emb = self.position_embedding(positions)

        x = item_emb + pos_emb

        # Dropout: 训练时随机置零部分特征
        x = self.dropout(x)

        # 将 padding 位置置 0
        valid_mask = (seq != 0).unsqueeze(-1)
        x = x*valid_mask

        # 创建 Mask，所有 Block 共用
        attn_mask = self.create_attention_mask(seq)

        # 依次执行多个 Transformer Block
        for block in self.blocks:
            # Attention 前的 LayerNorm
            q = block["attention_norm"](x)

            # Multi-Head Attention
            attn_output, _ = block["attention"](
                query=q,
                key=x,
                value=x,
                attn_mask=attn_mask,
                need_weights=False
            )

            # Attention 残差连接
            x = q + attn_output

            # FFN 前的 LayerNorm
            x = block["ffn_norm"](x)

            # FFN（内部已经包含残差连接）
            x = block["ffn"](x)

            # 将 Padding 位置重新置零
            x = x * valid_mask

        # 所有 Block 结束后，再进行最终 LayerNorm
        x = self.final_layer_norm(x)

        # 保证 Padding 位置最终为 0
        x = x * valid_mask

        return x