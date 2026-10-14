import torch
import torch.nn as nn

# 创建 LayerNorm 层
layer_norm = nn.LayerNorm(3)

# 输入一个三维向量
x = torch.tensor([[2.0, 4.0, 6.0]])

# 执行归一化
y = layer_norm(x)

print("输入：", x)
print("输出：", y)

print("gamma：", layer_norm.weight)
print("beta：", layer_norm.bias)
