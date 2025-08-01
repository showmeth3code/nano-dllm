import torch
from torch import nn
from torch.nn import functional as F
from nanovllm.layers.layernorm import RMSNorm

# 假设你的 RMSNorm 已经定义在当前作用域
# 这里我们用一个简单的输入向量 x 来演示：

x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)  # shape = [1, 4]
residual = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)

norm = RMSNorm(hidden_size=4) # 平方和的均值除以开方的结果为 2.7，然后 x / 2.7

# 标准 RMSNorm
out1 = norm(x)
print("→ 标准 RMSNorm 输出：")
print(out1)

x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)  # shape = [1, 4]
residual = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)
# 带残差的 RMSNorm（相当于 residual connection + norm）
out2, combined = norm(x, residual=residual)
print("\n→ 残差加法后 RMSNorm 输出：")
print(out2)
print("\n→ add 之后的 residual（即 x + residual）：")
print(combined)
