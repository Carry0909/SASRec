from data import load_data, split_data
from dataset import SASRecDataset
from model import SASRec
from train import train_model


def main():
    # 数据路径和超参数
    data_path = "data/Beauty.txt"

    maxlen = 50
    hidden_units = 64
    num_heads = 2
    num_blocks = 2
    dropout_rate = 0.2

    batch_size = 128
    epochs = 20
    learning_rate = 0.001

    # 加载用户行为序列
    users = load_data(data_path)
    if not users:
        raise ValueError("数据文件为空")

    # itemnum 是最大 item id，决定 embedding 表大小
    itemnum = max(item for items in users.values() for item in items)

    print("用户数量:", len(users))
    print("商品 ID 最大值:", itemnum)

    # 按时间切分：train / valid / test
    user_train, user_valid, user_test = split_data(users)

    # 只保留有足够交互的用户
    train_dataset = SASRecDataset(
        user_train=user_train,
        itemnum=itemnum,
        maxlen=maxlen,
    )

    print("可训练用户数量:", len(train_dataset))
    if len(train_dataset) == 0:
        raise ValueError("没有可用的训练样本")

    # 模型参数与 SASRec 论文保持一致
    model = SASRec(
        itemnum=itemnum,
        hidden_units=hidden_units,
        maxlen=maxlen,
        num_heads=num_heads,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
    )

    # 开始训练
    train_model(
        model=model,
        train_dataset=train_dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )


if __name__ == "__main__":
    main()