## 神经网络模型训练 20250316

import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from torch.optim.lr_scheduler import StepLR
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

# 加载数据集
faces = fetch_olivetti_faces(data_home='./data/Facedata', shuffle=True)
X = faces.data                  # 图像数据（每张图像已展平为 4096 维向量）
y = faces.target                # 标签（0-39，共40人）

# 按标签分层抽样拆分（确保每个人的图像均匀分布）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,              # 测试集比例
    random_state=42,            # 随机种子（确保可重复性）
    stratify=y                  # 按标签分层拆分
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__ (self):
        super(NeuralNetwork, self).__init__()
        self.flatten =nn.Flatten()
        self.linear_relu_stack= nn.Sequential(
            nn.Linear(4096, 8192),
            nn.BatchNorm1d(8192),                       #归一化
            nn.ReLU(),
            nn.Dropout(0.5),                            #正则化
            nn.Linear(8192, 16384),
            nn.BatchNorm1d(16384), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16384, 1024),
            nn.BatchNorm1d(1024), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 40) 
         )
    def forward(self, x): 
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

if __name__ == '__main__':
    model = NeuralNetwork().to(device)        

# 将 NumPy 数组转换为 PyTorch 张量，并转换为 float32 类型
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).long()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).long()

# 创建 Dataset 和 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128)


# 模型层分解,按层顺序构建模型
input_image=torch.rand(10,64,64)                        #取出小批量样本
flatten =nn.Flatten()                                   #将连续的维度范围展平成张量，即对输出进行处理，得到tensor类型的数据
flat_image = flatten(input_image)
layer1 = nn.Linear(in_features=64*64,out_features=10)   #定义并应用了一个全连接神经网络层用于处理图像数据，即对输入数据进行线性变换，进而输出一个新的变量
hidden1 = layer1(flat_image)
hidden1 = nn.ReLU()(hidden1)

seg_modules =nn.Sequential(                             #数据按照容器中定义的顺序（确保前一个模块输出大小和下一个模块输入大小保持一致）通过所有模块
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(10, 10)
)

input_image = torch.rand(10,64,64)                      #生成随机张量
logits = seg_modules(input_image)

softmax = nn.Softmax(dim=1)                             ## 神经网络的最后一个线性层返回的是logits类型的值，它们的取值是[-∞, ∞]。 把这些值传递给nn.Softmax模块。dim 参数指示我们在向量的哪个维度中计算softmax的值(和为1) 。
pred_probab = softmax(logits)

# 定制模型损失函数和优化器
loss_fn = nn.CrossEntropyLoss()                         #交叉熵损失函数
optimizer = torch.optim.AdamW(                          #AdamW优化器
                            model.parameters(),         #传入模型的可训练参数
                            lr=0.001,                   #学习率
                            betas=(0.9, 0.999),         #动量参数（默认）
                            eps=1e-08,                  #数值稳定性常数（默认）
                            weight_decay=0.01,          # **关键区别：权重衰减系数（建议0.01~0.1）**
                            amsgrad=False               # 是否使用AMSGrad变体（默认False）
                            )     

# 遍历数据集的每个批次（batch），计算模型预测值
def train(epoch,dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)                      #训练数据样本总量
    model.train()                                       #设置模型为训练模式
    
    for batch, (X, y) in enumerate(dataloader):         #遍历数据加载器的每个批次
        X, y = X.to(device), y.to(device)               #张量加载到设备
        #计算预测的误差
        pred = model(X)                                 #调用模型获得结果
        loss = loss_fn(pred, y)                         #计算损失
        #反向传播
        model.zero_grad()                               #重置模型中的梯度值为0
        loss.backward()                                 #计算梯度
        optimizer.step()                                #更新模型中参数的梯度值
 
        if batch % 100 == 0:
            loss_value = loss.item()
            current_samples = batch * len(X)
            print(f"Epoch {epoch}, Batch {batch}, Loss: {loss_value}, Samples: {current_samples}/{size}")
        
# 依赖测试数据集来检查模型
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)                      #获取数据集的总样本数
    num_batches = len(dataloader)                       #数据加载器的批次数，即总迭代次数
    model.eval()                                        #模型设置为评估模式
    test_loss, correct = 0, 0
    with torch.no_grad():                               #在测试过程中不跟踪梯度，减少内存消耗并加速计算
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)           
            pred = model(X)                             #模型前向传播，得到预测结果
            test_loss += loss_fn(pred, y).item()        #累加每个批次的损失值
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()         #统计预测正确的样本数
    test_loss /= num_batches                            #计算所有批次的平均损失
    correct /= size                                     #计算整体准确率（正确预测数 / 总样本数）
    
    print(f"Accuracy:{(100*correct):>0.2f}%, Avg loss:{test_loss:>10f}\n")

# 多轮训练
epochs = 30
for t in range(epochs):
    print(f"Epoch {t+1}: ")
    train(t, train_dataloader, model, loss_fn, optimizer)   # 传入 DataLoader
    test(test_dataloader, model, loss_fn)                   # 传入 DataLoader
print("训练完成!")
