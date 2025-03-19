import numpy as np
import pandas as pd
import matplotlib as plt

# Check the versions of the required libraries

print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Matplotlib version:", plt.__version__)

#第一周作业，使用cat函数拼接张量

import torch

# 创建两个张量
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])

# 沿着第0维（行）拼接
result = torch.cat((tensor1, tensor2), dim=0)
print("沿着第0维拼接的结果：\n", result)

# 沿着第1维（列）拼接
result = torch.cat((tensor1, tensor2), dim=1)
print("沿着第1维拼接的结果：\n", result)


