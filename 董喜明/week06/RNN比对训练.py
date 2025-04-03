# --------------------------------------
# 实验使用不同的RNN结构，实现一个人脸图像分类器。至少对比2种以上结构训练损失和准确率差异，如：LSTM、GRU、RNN、BiRNN等。要求使用tensorboard，提交代码及run目录和可视化截图
# 时间：20250402
# --------------------------------------

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime

# 配置计算设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------------------
# 数据预处理
# --------------------------------------
faces = fetch_olivetti_faces(data_home='./data/Facedata')   # 加载数据集（移除shuffle参数）

# 将图像数据reshape为3D张量 (样本数, 序列长度, 特征维度)
X = faces.data.reshape(-1, 64, 64)                          # 原始图像尺寸64x64
y = faces.target  

# 拆分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
    shuffle=True                                            # 在此处设置shuffle
)

# 转换为PyTorch张量
X_train_tensor = torch.from_numpy(X_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_train_tensor = torch.from_numpy(y_train).long()
y_test_tensor = torch.from_numpy(y_test).long()

# 创建Dataset和DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64)

# --------------------------------------
# 模型定义
# --------------------------------------
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, dropout=0.5, num_layers=2)
        self.fc = nn.Linear(hidden_size, num_classes, bias=True)

    def forward(self, x):
        # x形状: (batch_size, seq_len, input_size)
        out, _ = self.rnn(x)
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        return self.fc(out)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.5, num_layers=2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, dropout=0.5, num_layers=2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

# --------------------------------------
# 训练配置
# --------------------------------------
input_size = 64                     # 每个时间步的特征维度
hidden_size = 256                   # 隐藏层大小
num_classes = 40                    # 类别数
num_epochs = 60                     # 训练轮数 
learning_rate = 0.001               # 学习率

# 初始化模型
models = {
    "RNN": RNNModel(input_size, hidden_size, num_classes).to(device),
    "LSTM": LSTMModel(input_size, hidden_size, num_classes).to(device),
    "GRU": GRUModel(input_size, hidden_size, num_classes).to(device)
}

# 定义训练函数
def train_model(model, train_loader, test_loader, writer, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4, amsgrad=True)
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 记录训练损失
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Training Loss', avg_loss, epoch)
        
        # 测试阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        writer.add_scalar('Test Accuracy', accuracy, epoch)
        
        print(f"{model_name} Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# --------------------------------------
# 训练所有模型
# --------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")    # 在循环外生成统一时间戳（保证同次运行不同模型使用相同时间标记）

for name, model in models.items():
    # 创建包含时间戳的独立日志目录
    log_dir = f"runs/{timestamp}_{name}"
    writer = SummaryWriter(log_dir)
    
    print(f"Training {name} [{timestamp}]...")
    train_model(model, train_dataloader, test_dataloader, writer, name)
    writer.close()
