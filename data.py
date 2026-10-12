def load_data(file_path):
    """ 读取数据，把每个用户互动序列组成列表，最后组成字典 """
    users = {}
    with open(file_path, 'r') as f:
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
    """ 将用户字典划分训练集、验证集、测试集 """
    user_train = {}
    user_valid = {}
    user_test = {}

    for user, items in users.items():
        if len(items) < 4:
            # 交互数少于4，全部保留在训练集
            user_train[user] = items
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = items[:-2]
            user_valid[user] = [items[-2]]
            user_test[user] = [items[-1]]

    return user_train, user_valid, user_test

if __name__ == "__main__":
    users_dict = load_data("sample.txt")

    user_train, user_valid, user_test = split_data(users_dict)

    print("训练集：", user_train)
    print("验证集：", user_valid)
    print("测试集：", user_test)

    assert set(user_train) == set(user_valid) == set(user_test)
    print("数据划分测试通过！")