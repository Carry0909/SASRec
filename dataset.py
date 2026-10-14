import random
import torch
from torch.utils.data import Dataset


def build_sequence(items, maxlen):
    """构造 SASRec 输入序列 seq 和正样本 pos"""

    # 输入序列和正样本错开一个位置
    seq = items[:-1]
    pos = items[1:]

    # 只保留最近 maxlen 个位置
    if len(seq) > maxlen:
        seq = seq[-maxlen:]
        pos = pos[-maxlen:]

    # 左侧补零
    if len(seq) < maxlen:
        pad = maxlen - len(seq)
        seq = [0] * pad + seq
        pos = [0] * pad + pos

    return seq, pos


def sample_negative(items, itemnum, pos):
    """为每个有效预测位置采样一个负样本"""

    item_set = set(items)

    if len(item_set) >= itemnum:
        raise ValueError("没有可用的负样本")

    neg = [0] * len(pos)

    for i in range(len(pos)):
        if pos[i] == 0:
            continue

        while True:
            item_id = random.randint(1, itemnum)

            if item_id not in item_set:
                neg[i] = item_id
                break

    return neg


class SASRecDataset(Dataset):

    def __init__(self, user_train, itemnum, maxlen):
        self.user_train = user_train
        self.itemnum = itemnum
        self.maxlen = maxlen

        # 只保留至少有两个训练交互的用户
        self.user_ids = []

        for uid, items in user_train.items():
            if len(items) >= 2:
                self.user_ids.append(uid)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        # 取得用户交互序列
        uid = self.user_ids[index]
        items = self.user_train[uid]

        # 构造训练样本
        seq, pos = build_sequence(items, self.maxlen)
        neg = sample_negative(items, self.itemnum, pos)

        # 转为 PyTorch Tensor
        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long)
        )