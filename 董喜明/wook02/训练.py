## 1.训练

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载数据并取前100个样本
X, y = load_iris(return_X_y=True)
X = X[:100]
y = y[:100]

# 拆分数据集，区分训练集和测试集（可调整test_size参数）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化参数
n_features = X_train.shape[1]
w = np.zeros(n_features)
b = 0.0

# 定义sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 超参数设置（可调整learning_rate）
learning_rate = 0.1
n_iters = 1000

# 训练循环
for i in range(n_iters):
    # 前向传播
    z = np.dot(X_train_scaled, w) + b
    y_pred = sigmoid(z)
    y_pred = np.clip(y_pred, 1e-15, 1-1e-15)  # 防止log(0)
    
    # 计算损失
    loss = -np.mean(y_train * np.log(y_pred) + (1 - y_train) * np.log(1 - y_pred))
    
    # 计算梯度
    dw = (1 / len(y_train)) * np.dot(X_train_scaled.T, (y_pred - y_train))
    db = (1 / len(y_train)) * np.sum(y_pred - y_train)
    
    # 更新参数
    w -= learning_rate * dw
    b -= learning_rate * db
    
    # 每100次打印损失
    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {loss:.4f}")

# 评估模型
def predict(X, w, b):
    z = np.dot(X, w) + b
    return (sigmoid(z) >= 0.5).astype(int)

train_acc = np.mean(predict(X_train_scaled, w, b) == y_train)
test_acc = np.mean(predict(X_test_scaled, w, b) == y_test)
print(f"Train Accuracy: {train_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")

# 保存参数和标准化信息
np.savez("model_params.npz", 
         w=w, 
         b=b, 
         mean=scaler.mean_, 
         scale=scaler.scale_)
