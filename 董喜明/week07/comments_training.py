# ------------------------------------------------------
# 使用豆瓣电影评论数据完成文基于深度学习的文本分类RNN模型实现。
# 时间：20250416
# ------------------------------------------------------

import csv
import jieba
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split 

# ------------------------------------------------------
# 将原始评论数据序列化，完成文本分类处理：文本预处理后构建词典。（评论得分1～2代表negative取值：0，评论得分4～5代表positive取值：1）
# ------------------------------------------------------

# 1. 定义用户评论数据集
ds_comments = []

# 2. 加载停用词表
stopwords_path = 'data/Douban/stopwords.txt' 
stopwords = set()
try:
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip())
    print(f"已加载停用词数量：{len(stopwords)}")
except FileNotFoundError:
    print(f"警告：停用词文件 {stopwords_path} 未找到，跳过停用词过滤")

# 3. 读取原始评论文件，获取评论文本和对应分值
with open('data/Douban/comments.csv', 'r', encoding='utf-8') as file: 
    reader = csv.DictReader(file)                                          
    for row in reader:                                                    
        vote = int(row['votes'])                                    
        if vote in [1, 2, 4, 5]:                                        # 筛选评分为1/2/4/5的评论                                        
            words = jieba.lcut(row['content'])                            
            words = [word for word in words if word.strip()]            # 去空白词             
            label = 1 if vote in [4, 5] else 0                          # 高分(4-5)为正向(1)，低分(1-2)为负向(0)
            ds_comments.append((words, label)) 

print(f"原始评论数量：{len(ds_comments)}")

# 4. 过滤掉停用词
ds_comments = [(list(filter(lambda x: x not in stopwords, c)), v) for c,v in ds_comments]       # 过滤掉停用词
print(f"过滤停用词后评论数量：{len(ds_comments)}")

# 5. 过滤过短、过长的无效评论
ds_comments = [c for c in ds_comments if len(c[0]) in range(2, 200)]       # 过滤掉长度小于2或大于200的评论
print(f"过滤评论长度后评论数量：{len(ds_comments)}")

# 6. 统计评论长度分布
comments_len = [len(c) for c,v in ds_comments]                              # 统计评论长度分布
plt.hist(comments_len, bins=100)
print(f"评论长度均值：{sum(comments_len)/len(comments_len):.2f}")

# 7. 统计正负样本比例
positive_count = sum(1 for _, label in ds_comments if label == 1)
negative_count = len(ds_comments) - positive_count
# print(f"正负样本比例: {positive_count}:{negative_count}")

# 8. 划分训练、测试集
train_data, test_data = train_test_split(
    ds_comments, 
    test_size=0.2, 
    stratify=[d[1] for d in ds_comments], 
    random_state=42
)
print(f"训练集数量：{len(train_data)}")
print(f"测试集数量：{len(test_data)}")

# 9. 保存处理后的数据集
# with open('data/Douban/comments_train.pkl', 'wb') as f:
#     pickle.dump(train_data, f)
# with open('data/Douban/comments_test.pkl', 'wb') as f:
#     pickle.dump(test_data, f)

# ------------------------------------------------------
# 加载处理后文本构建词典、定义、训练、保存模型
# ------------------------------------------------------

# 1. 构建词典，将文本中的每个单词映射到一个唯一的索引
def build_from_doc(doc):
    vocab = set()
    for line in doc:
        vocab.update(line[0])

    vocab =  ['PAD','UNK'] + list(vocab)       
    w2idx = {word: idx for idx, word in enumerate(vocab)}
    return w2idx

# 2. 定义模型（LSTM）
class Comments_Classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        output, (hidden, _) = self.rnn(embedded)
        output = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(output)

# 3. 训练模型
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3.1 加载训练集和测试集
    with open('data/Douban/comments_train.pkl','rb') as f:
        train_data = pickle.load(f)
    with open('data/Douban/comments_test.pkl','rb') as f:
        test_data = pickle.load(f)

    # 3.2 构建词汇表应仅使用训练集
    vocab = build_from_doc(train_data)
    print('词汇表大小:', len(vocab))

    # 3.3 所有向量集合 Embedding（词嵌入）
    emb = nn.Embedding(len(vocab), 100) # 词汇表大小，向量维度
    print('词汇表向量:', emb.weight.shape)

    # 3.4 定义数据加载器
    def convert_data(batch_data):
        comments, votes = [],[]
        # 分别提取评论和标签
        for comment, vote in batch_data:
            comments.append(torch.tensor([vocab.get(word, vocab['UNK']) for word in comment]))
            votes.append(vote)
            # print(comments)
            # print(votes)
            # break
        
        # 填充为相同长度
        commt = pad_sequence(comments, batch_first=True, padding_value=vocab['PAD'])
        labels = torch.tensor(votes)
        # 返回评论和标签
        return commt, labels

    # 3.5 加载数据
    train_loader = DataLoader(train_data, 
                            batch_size=16, 
                            shuffle=True, 
                            collate_fn=convert_data)
    
    test_loader = DataLoader(test_data,
                           batch_size=16,
                           shuffle=False,
                           collate_fn=convert_data)

    # 3.6 定义模型超参数
    vocab_size = len(vocab)             # 词汇表大小
    embedding_dim = 256                 # 词嵌入维度
    hidden_size = 256                   # LSTM隐藏层大小
    num_classes = 2                     # 分类数量
    num_epochs = 10                     # 训练轮数

    # 3.7 实例化模型
    model = Comments_Classifier(len(vocab), embedding_dim, hidden_size, num_classes)
    model.to(device)

    # 3.8 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 3.8 添加测试集评估函数
    def evaluate(model, dataloader):
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for cmt, lbl in dataloader:
                cmt = cmt.to(device)
                lbl = lbl.to(device)
                outputs = model(cmt)
                _, preds = torch.max(outputs, 1)
                total += lbl.size(0)
                correct += (preds == lbl).sum().item()
        return correct / total

    # # 3.9 训练模型
    # best_acc = 0.0
    # for epoch in range(num_epochs):
    #     model.train()
    #     for i, (cmt, lbl) in enumerate(train_loader):
    #         cmt = cmt.to(device)
    #         lbl = lbl.to(device)

    #         # 前向传播
    #         outputs = model(cmt)
    #         loss = criterion(outputs, lbl)

    #         # 反向传播和优化
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         if (i+1) % 10 == 0:
    #             test_acc = evaluate(model, test_loader)
    #             print(f"Epoch [{epoch+1}/{num_epochs}], Test Acc: {test_acc:.4f}")

    # # 3.10 保存模型
    # if test_acc > best_acc:
    #         best_acc = test_acc
    #         torch.save(model.state_dict(), 'model/comments_training.model')
    #         torch.save(vocab, 'model/comments_training.vocab')

# ------------------------------------------------------
# 加载训练后保存的模型和词典进行测试
# ------------------------------------------------------

    # 1. 加载词典
    vocab = torch.load('model/comments_training.vocab')

    # 2. 测试模型
    comment1 = '这部电影的视觉效果简直令人震撼，剧本深刻且演员表演出色，近几年难得的好片，绝对值得五星推荐！'
    comment2 = '剧情拖沓毫无逻辑，演员演技生硬，是怎么选的角啊，完全浪费时间的烂片，差评！'

    # 3. 将评论转换为索引
    comment1_idx = torch.tensor([vocab.get(word, vocab['UNK']) for word in jieba.lcut(comment1)])
    comment2_idx = torch.tensor([vocab.get(word, vocab['UNK']) for word in jieba.lcut(comment2)])
    # print(comment1_idx)
    # print(comment2_idx) 


    # 4. 将评论转换为tensor
    comment1_idx = comment1_idx.unsqueeze(0).to(device)  
    comment2_idx = comment2_idx.unsqueeze(0).to(device)  
    # print(comment1_idx)
    # print(comment2_idx)

    # 5. 加载模型
    model = Comments_Classifier(len(vocab), embedding_dim, hidden_size, num_classes)
    model.load_state_dict(torch.load('model/comments_training.model'))
    model.to(device)
    final_acc = evaluate(model, test_loader)
    print(f"最终测试集准确率: {final_acc:.4f}")

    # 6. 模型推理
    pred1 = model(comment1_idx)
    pred2 = model(comment2_idx)
    print(pred1)
    print(pred2)

    # 7. 取最大值的索引作为预测结果
    pred1 = torch.argmax(pred1, dim=1).item()
    pred2 = torch.argmax(pred2, dim=1).item()
    print(f'评论1预测结果: {pred1}')
    print(f'评论2预测结果: {pred2}')
