
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def calculate_loss(pos_logits, neg_logits, pos):
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    pos_loss = criterion(
        pos_logits,
        torch.ones_like(pos_logits)
    )

    neg_loss = criterion(
        neg_logits,
        torch.zeros_like(neg_logits)
    )

    # 排除 Padding 位置
    valid_mask = (pos != 0)

    loss = (pos_loss + neg_loss)[valid_mask].mean()

    return loss


def train_model(model, train_dataset, epochs=10,
                batch_size=128, learning_rate=0.001):

    # 1. 选择设备
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    # 2. 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # 3. 创建优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # 4. Epoch 循环
    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        num_batches = 0

        # 5. Batch 循环
        for seq, pos, neg in train_loader:

            # 将数据移动到模型所在设备
            seq = seq.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            # 排除完全没有有效目标的 Batch
            if not (pos != 0).any().item():
                continue

            # 清空上一次的梯度
            optimizer.zero_grad()

            # 前向传播
            pos_logits, neg_logits = model.forward_train(
                seq, pos, neg
            )

            # 计算损失
            loss = calculate_loss(pos_logits, neg_logits, pos)

            # 反向传播
            loss.backward()

            # 更新模型参数
            optimizer.step()

            # 记录损失
            total_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            raise ValueError("训练集中没有有效的预测位置")

        avg_loss = total_loss / num_batches

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
