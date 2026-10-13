def load_data(file_path):
    """读取数据，将每个用户的交互记录整理成字典"""
    users = {}

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            user = int(parts[0])
            item = int(parts[1])

            if user not in users:
                users[user] = []

            users[user].append(item)

    return users


def split_data(users):
    """划分训练集、验证集、测试集"""
    user_train = {}
    user_valid = {}
    user_test = {}

    for user, items in users.items():
        if len(items) < 4:
            user_train[user] = items
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = items[:-2]
            user_valid[user] = [items[-2]]
            user_test[user] = [items[-1]]

    return user_train, user_valid, user_test