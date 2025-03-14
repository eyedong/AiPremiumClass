## 1.数据预处理调试 20250312

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from datetime import datetime
from torch.optim.lr_scheduler import StepLR

# 1.1训练集数据保存到本地

#training_data = datasets.FashionMNIST(
#    root='data',
#    train=True,
#    download=True,
#    transform=ToTensor(),
#)
training_data = datasets.KMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),
)

# 1.2测试集数据保存到本地

#test_data = datasets.FashionMNIST(
#    root='data',
#    train=False,
#    download=True,
#    transform=ToTensor(),
#)
test_data = datasets.KMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor(),
)

#  1.3定义单一批次的数据样本量大小

batch_size = 64

#  1.4创建数据加载器

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

#  1.5测试数据加载器输出

for X,y in test_dataloader:
    print ("Shape of X [N,C,H,W]:",X.shape)
    print ("Shape of y:",y.shape,y.dtype)
    break


## 2.构建模型调试 20250312

# 2.1检验可以使用的设备

device ="cuda" if torch.cuda.is_available() else "cpu"
print(f"使用 {device} 设备")

# 2.2定义神经网络模型

class NeuralNetwork(nn.Module):                     #通过继承nn.Module父类来实现自定义的网络模型
    def __init__ (self):                            #初始化神经网络层
        super(NeuralNetwork, self).__init__()
        self.flatten =nn.Flatten()
        self.linear_relu_stack= nn.Sequential(
            nn.Linear(28*28, 512),                  #第一层神经元数512
            nn.ReLU(),
            nn.Linear(512, 256),                    #第二层神经元数256
            nn.ReLU(),
            nn.Linear(256, 128),                    #第三层神经元数128
            nn.ReLU(),
            nn.Linear(128, 10)                      #第四层神经元数10
        )
    def forward(self, x):                           #对输入数据进行操作
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)        
print(model)

# 2.3模型层分解

# 取一个由3张大小为28x28的图像的小批量样本
input_image=torch.rand(3,28,28)
print(input_image.size())

# 将每个28x28大小的二维图像转换为784个像素值的连续数组
flatten =nn.Flatten()                               #初始化nn.Flatten层
flat_image = flatten(input_image)
print(flat_image.size())

# 定义并应用了一个全连接神经网络层（线性变换），用于处理图像数据
layer1 = nn.Linear(in_features=28*28,out_features=20)   #输入：一张 28x28 的图像被展平为 [784] 的 1D 向量，通过全连接层将 784 维映射到 20 维（线性变换）得到形状为 [20] 的新特征向量
hidden1 = layer1(flat_image)
print(hidden1.size())

print(f"ReLU 之前的数据:{hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"ReLU 之后的数据:{hidden1}")

# nn.Sequential是一个有序的模块容器，数据按照容器中定义的顺序通过所有模块（快速处理网络）
seg_modules =nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)

input_image = torch.rand(3,28,28)
logits = seg_modules(input_image)

# 神经网络的最后一个线性层返回的是logits类型的值，它们的取值是[-∞, ∞]。 把这些值传递给nn.Softmax模块
softmax = nn.Softmax(dim=1)                     #dim参数指示我们在向量的哪个维度中计算softmax的值(和为1) 。
pred_probab = softmax(logits)


## 3.模型参数调试 20250312

# 3.1 迭代模型中的参数

print("Model structure:",model,"\n\n")
for name,param in model.named_parameters():
    print(f"Layer: {name}|Size:{param.size()} | Values : {param[:2]} \n")

# 3.2 定制模型损失函数和优化器

loss_fn = nn.CrossEntropyLoss()                 #交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(), #使用随机梯度下降方法的优化器
                            lr=0.05,            #学习率
                            momentum=0.9        #momentum参数用于加速梯度降并减少震荡
                            )    

# 3.3 在单个训练循环中，模型对训练数据集进行预测（分批输入），并反向传播预测误差以调整模型参数

# history store obs
his_loss = []
his_epoch = []
his_acc = []

# 遍历数据集的每个批次（batch），计算模型预测值
def train(epoch,dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)          #训练数据样本总量
    model.train()                           #设置模型为训练模式
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)   #张量加载到设备

        #计算预测的误差
        pred = model(X)                     #调用模型获得结果
        loss = loss_fn(pred, y)             #计算损失

        #反向传播
        model.zero_grad()                   #重置模型中的梯度值为0
        loss.backward()                     #计算梯度
        optimizer.step()                    #更新模型中参数的梯度值
 
        if batch % 100 == 0:
            loss, corrent = loss.item(), batch * len (X)
            print(f"loss:{loss:7f} [{corrent:>5d}/{size:>5d}]")
        
# 3.4 依赖测试数据集来检查模型

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()                            #模型设置为评估模式
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test error: \n Accuracy:{(100*correct):>0.2f}%, Avg loss:{test_loss:>10f}\n")

# 3.5 训练过程在多轮迭代，每次模型通过学习更新内置的参数，以便做出更好的预测

epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n------------------------------------")
    train(t,train_dataloader, model, loss_fn, optimizer)
#    if t % 10 == 0:
    test(test_dataloader, model, loss_fn)
print("训练完成!")
