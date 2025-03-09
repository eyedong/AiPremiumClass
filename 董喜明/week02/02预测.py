## 2.预测

import numpy as np

# 加载参数
params = np.load("model_params.npz")
w = params["w"]
b = params["b"]
mean = params["mean"]
scale = params["scale"]

# 输入新数据（示例）
X_new = np.array([[5.1, 3.5, 1.4, 0.2]])  # 替换为实际数据

# 标准化
X_new_scaled = (X_new - mean) / scale

# 预测
z = np.dot(X_new_scaled, w) + b
probability = 1 / (1 + np.exp(-z))
pred_class = 1 if probability > 0.5 else 0

print(f"Predicted class: {pred_class}, Probability: {probability[0]:.4f}")
