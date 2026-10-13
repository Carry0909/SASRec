from torch.utils.data import DataLoader

from data import load_data, split_data
from dataset import SASRecDataset


# 1. 读取原始数据
users = load_data("sample.txt")
print("原始用户数据：", users)


# 2. 划分数据集
user_train, user_valid, user_test = split_data(users)

print("\n训练集：", user_train)
print("验证集：", user_valid)
print("测试集：", user_test)

assert set(user_train) == set(user_valid) == set(user_test)
print("数据划分测试通过！")


# 3. 创建 Dataset
# 前提：物品 ID 连续编号为 1 到 itemnum
itemnum = max(
    item
    for items in users.values()
    for item in items
)

dataset = SASRecDataset(
    user_train=user_train,
    itemnum=itemnum,
    maxlen=5
)

print("\nDataset 长度：", len(dataset))


# 4. 测试单条训练样本
if len(dataset) > 0:
    seq, pos, neg = dataset[0]

    print("\n第一条训练样本：")
    print("seq:", seq)
    print("pos:", pos)
    print("neg:", neg)


# 5. 创建并测试 DataLoader
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

for seq, pos, neg in loader:
    print("\nDataLoader 测试：")
    print("seq:\n", seq)
    print("pos:\n", pos)
    print("neg:\n", neg)

    print("seq shape:", seq.shape)
    print("pos shape:", pos.shape)
    print("neg shape:", neg.shape)
    break