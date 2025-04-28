# ------------------------------------------------------
# 使用中文对联数据集(Chinese-Couplets-Dataset)训练带有attention的seq2seq模型。
# 时间：20250427
# ------------------------------------------------------

import os
import re
import pickle
import jieba
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# ==================== 数据预处理 ====================

def process_couplets_dataset(in_path, out_path, 
                            in_save_path, out_save_path,
                            stopwords_path,
                            min_len=5, max_len=50):
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(in_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_save_path), exist_ok=True)

    # 加载停用词表
    if not os.path.exists(stopwords_path):
        raise FileNotFoundError(f"停用词文件不存在：{stopwords_path}")
    
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f])
    
    # 中文过滤正则表达式（保留中文、数字、常用标点）
    chinese_pattern = re.compile(
        r'[^\u4e00-\u9fa5，。！？；：“”‘’（）《》【】、\d\s]'
    )

    def clean_text(text):
        """文本清洗函数"""
        # 去除特殊字符
        text = re.sub(chinese_pattern, '', text)
        # 合并空白字符
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def tokenize(text):
        """分词处理函数"""
        # 精确模式分词
        words = jieba.lcut(text)
        # 过滤停用词和空白
        return [w for w in words if w not in stopwords and w.strip() != '']

    # 处理数据
    processed_in = []
    processed_out = []

    try:
        with open(in_path, 'r', encoding='utf-8') as f_in, \
             open(out_path, 'r', encoding='utf-8') as f_out:

            total_lines = sum(1 for _ in f_in)
            f_in.seek(0), f_out.seek(0)

            for in_line, out_line in tqdm(zip(f_in, f_out), 
                                        total=total_lines,
                                        desc="处理对联数据"):
                # 清洗文本
                in_clean = clean_text(in_line.strip())
                out_clean = clean_text(out_line.strip())

                # 长度过滤
                if not (min_len <= len(in_clean) <= max_len):
                    continue
                if not (min_len <= len(out_clean) <= max_len):
                    continue

                # 分词处理
                in_tokens = tokenize(in_clean)
                out_tokens = tokenize(out_clean)

                # 有效性检查
                if len(in_tokens) == 0 or len(out_tokens) == 0:
                    continue
                if abs(len(in_tokens) - len(out_tokens)) > 2:
                    continue  # 保持上下联长度平衡

                processed_in.append(in_tokens)
                processed_out.append(out_tokens)

    except FileNotFoundError as e:
        print(f"文件不存在：{e}")
        return

    # 保存处理结果
    with open(in_save_path, 'wb') as f:
        pickle.dump(processed_in, f)
    
    with open(out_save_path, 'wb') as f:
        pickle.dump(processed_out, f)

    # 打印处理结果
    print(f"\n处理完成，共保留 {len(processed_in)} 对有效对联")
    print("样例数据：")
    print("上联：", " ".join(processed_in[0]))
    print("下联：", " ".join(processed_out[0]))

if __name__ == "__main__":
    # 配置参数
    config = {
        "in_path": "data/Couplets/fixed_couplets_in.txt",
        "out_path": "data/Couplets/fixed_couplets_out.txt",
        "in_save_path": "data/Couplets/fixed_couplets_in.pkl",
        "out_save_path": "data/Couplets/fixed_couplets_out.pkl",
        "stopwords_path": "data/Douban/stopwords.txt",
        "min_len": 5,   # 最小保留字数
        "max_len": 50    # 最大保留字数
    }

    # 运行处理流程
    process_couplets_dataset(**config)

# ==================== 词汇表构建 ====================
class Vocabulary:
    def __init__(self):
        self.char2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2char = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.count = 4

    def build_vocab(self, tokenized_data, min_freq=2):
        counter = Counter()
        for tokens in tokenized_data:
            counter.update(tokens)
        
        for token, freq in counter.items():
            if freq >= min_freq:
                self.char2idx[token] = self.count
                self.idx2char[self.count] = token
                self.count += 1

# ==================== 数据集处理 ====================
class CoupletDataset(Dataset):
    def __init__(self, in_path, out_path, vocab, max_len=50):
        with open(in_path, 'rb') as f:
            self.in_data = pickle.load(f)
        with open(out_path, 'rb') as f:
            self.out_data = pickle.load(f)
        # 确保输入和输出长度一致
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.in_data)

    def __getitem__(self, idx):
        in_seq = [self.vocab.char2idx.get(t, 3) for t in self.in_data[idx]]
        out_seq = [self.vocab.char2idx.get(t, 3) for t in self.out_data[idx]]
        
        # 添加EOS并填充
        in_seq = in_seq[:self.max_len-1] + [2] 
        in_seq += [0]*(self.max_len - len(in_seq))
        # 添加SOS和EOS并填充
        out_seq = [1] + out_seq[:self.max_len-2] + [2]
        out_seq += [0]*(self.max_len - len(out_seq))
        # 转换为张量
        return torch.LongTensor(in_seq), torch.LongTensor(out_seq)

# ==================== 模型定义 ====================
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        # 使用LSTM，bidirectional=True表示双向LSTM
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, bidirectional=True)
        # 合并双向状态
        self.fc = nn.Linear(hid_dim*2, hid_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        # 使用LSTM，bidirectional=True表示双向LSTM
        outputs, (hidden, cell) = self.rnn(embedded)
        # 合并双向状态
        hidden = self.fc(torch.cat((hidden[0::2], hidden[1::2]), dim=2))
        # 合并双向状态
        cell = self.fc(torch.cat((cell[0::2], cell[1::2]), dim=2))
        return outputs, (hidden, cell)

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        # 使用线性层计算注意力权重
        self.attn = nn.Linear(hid_dim * 3, hid_dim)
        # 使用线性层计算注意力权重
        self.v = nn.Linear(hid_dim, 1)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.shape[0]
        # 重复隐藏状态以匹配编码器输出的序列长度
        hidden = hidden.repeat(seq_len, 1, 1)
        # 计算注意力权重
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # 计算注意力权重
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=0)

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.attention = Attention(hid_dim)
        self.rnn = nn.LSTM(emb_dim + hid_dim*2, hid_dim, n_layers)
        self.fc = nn.Linear(hid_dim*3 + emb_dim, vocab_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        x = x.unsqueeze(0)  # [1, batch_size]
        embedded = self.embedding(x)  # [1, batch_size, emb_dim]
        # 计算注意力权重
        attn_weights = self.attention(hidden[-1], encoder_outputs)  # [seq_len, batch_size]
        # 修正上下文计算
        context = torch.einsum("sb,sbd->bd", attn_weights, encoder_outputs)  # [batch_size, hid_dim*2]
        context = context.unsqueeze(0)  # [1, batch_size, hid_dim*2]
        # 拼接输入
        rnn_input = torch.cat((embedded, context), dim=2)  # [1, batch_size, emb_dim+hid_dim*2]
        # RNN计算
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))  # output: [1, batch_size, hid_dim]
        # 拼接输出特征
        output = output.squeeze(0)  # [batch_size, hid_dim]
        context = context.squeeze(0)  # [batch_size, hid_dim*2]
        embedded = embedded.squeeze(0)  # [batch_size, emb_dim]
        # 拼接输出特征
        combined = torch.cat((output, context, embedded), dim=1)  # [batch_size, hid_dim*3 + emb_dim]
        prediction = self.fc(combined)
        return prediction, hidden, cell

# ==================== 训练配置 ====================
def train_model():
    # 加载数据并构建词汇表
    vocab = Vocabulary()
    with open('data/Couplets/fixed_couplets_in.pkl', 'rb') as f:
        in_data = pickle.load(f)
    with open('data/Couplets/fixed_couplets_out.pkl', 'rb') as f:
        out_data = pickle.load(f)
    
    vocab.build_vocab(in_data + out_data)
    # 打印词汇表
    print(f"词汇表大小: {len(vocab.char2idx)}")

    # 创建完整数据集
    full_dataset = CoupletDataset(
        'data/Couplets/fixed_couplets_in.pkl',
        'data/Couplets/fixed_couplets_out.pkl',
        vocab
    )

    # 打印数据集
    print(f"数据集大小: {len(full_dataset)}")

    # 划分训练集和测试集
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )
    # 打印训练集和测试集
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=256, num_workers=4, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, num_workers=4, pin_memory=True, shuffle=False)
    
    # 打印加载器
    print(f"训练集加载器大小: {len(train_loader)}")
    print(f"测试集加载器大小: {len(test_loader)}")
    # 打印示例
    for src, trg in train_loader:
        print(f"输入形状: {src.shape}")  # 应为 [seq_len, batch_size]
        print(f"输出形状: {trg.shape}")  # 应为 [seq_len, batch_size]
        break

    # 模型参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(len(vocab.char2idx), 512, 1024).to(device)
    decoder = Decoder(len(vocab.char2idx), 512, 1024).to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # 打印模型参数数量
    print(f"编码器参数数量: {sum(p.numel() for p in encoder.parameters() if p.requires_grad)}")
    print(f"解码器参数数量: {sum(p.numel() for p in decoder.parameters() if p.requires_grad)}")

    # 训练循环
    best_test_loss = float('inf')
    for epoch in range(5):
        print(f"开始第 {epoch+1} 轮训练...")
        # ======== 训练阶段 ========
        encoder.train()
        decoder.train()
        train_loss = 0
        for src, trg in train_loader:
            print("处理批次中...")
            src = src.transpose(0, 1).to(device)
            trg = trg.transpose(0, 1).to(device)
            
            optimizer.zero_grad()
            encoder_outputs, (hidden, cell) = encoder(src)
            
            loss = 0
            for t in range(1, trg.size(0)):
                predictions, hidden, cell = decoder(trg[t-1], hidden, cell, encoder_outputs)
                loss += criterion(predictions, trg[t])
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # ======== 验证阶段 ========
        encoder.eval()
        decoder.eval()
        test_loss = 0
        with torch.no_grad():
            for src, trg in test_loader:
                src = src.transpose(0, 1).to(device)
                trg = trg.transpose(0, 1).to(device)
                
                encoder_outputs, (hidden, cell) = encoder(src)
                
                loss = 0
                for t in range(1, trg.size(0)):
                    predictions, hidden, cell = decoder(trg[t-1], hidden, cell, encoder_outputs)
                    loss += criterion(predictions, trg[t])
                
                test_loss += loss.item()

        # 打印统计信息
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        print(f'Epoch {epoch}')
        print(f'Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}')

        # 保存最佳模型
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'vocab': vocab,
                'test_loss': avg_test_loss
            }, 'best_couplet_model.pth')
            print("发现更好模型，已保存！")

    # 保存最终模型
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'vocab': vocab
    }, 'model/couplet_model.pth')

if __name__ == '__main__':
    train_model()

# ==================== 模型测试 ====================
class CoupletGenerator:
    def __init__(self, model_path, max_len=50):
        # 加载保存的模型和词典
        checkpoint = torch.load('model/couplet_model.pth', map_location='cpu')
        self.vocab = checkpoint['vocab']
        self.max_len = max_len
        
        # 初始化模型
        self.encoder = Encoder(len(self.vocab.char2idx), 512, 1024)
        self.decoder = Decoder(len(self.vocab.char2idx), 512, 1024)
        
        # 加载模型参数
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        
        # 设置为评估模式
        self.encoder.eval()
        self.decoder.eval()
    
    def preprocess(self, input_str):
        """预处理输入对联"""
        # 转换为字符列表
        tokens = list(input_str.strip())
        # 转换为索引
        indices = [self.vocab.char2idx.get(t, 3) for t in tokens]
        # 添加EOS并填充
        indices = indices[:self.max_len-1] + [2] 
        indices += [0]*(self.max_len - len(indices))
        return torch.LongTensor(indices).unsqueeze(1).to(self.device)  # [seq_len, 1]
    
    def generate(self, input_str):
        # 预处理输入
        input_tensor = self.preprocess(input_str)
        
        with torch.no_grad():
            # 编码阶段
            encoder_outputs, (hidden, cell) = self.encoder(input_tensor)
            
            # 解码阶段
            batch_size = input_tensor.shape[1]
            # 初始化解码器输入（SOS标记）
            decoder_input = torch.LongTensor([1]).to(self.device).unsqueeze(1)  # [1, 1]
            
            output_indices = []
            for _ in range(self.max_len):
                predictions, hidden, cell = self.decoder(
                    decoder_input, hidden, cell, encoder_outputs
                )
                # 获取当前时间步预测的token
                pred_token = predictions.argmax(1).item()
                output_indices.append(pred_token)
                
                # 遇到EOS停止生成
                if pred_token == 2:
                    break
                
                # 准备下一步输入
                decoder_input = torch.LongTensor([pred_token]).to(self.device).unsqueeze(1)
            
            # 转换为字符
            output_str = ''.join([
                self.vocab.idx2char.get(idx, '<UNK>') 
                for idx in output_indices
                if idx not in [0, 1, 2]  # 过滤特殊标记
            ])
            
        return output_str

# ==================== 模型验证 ====================
if __name__ == '__main__':
    # 初始化生成器
    generator = CoupletGenerator('model/couplet_model.pth')
    
    # 交互式对联生成
    while True:
        input_str = input("\n请输入上联（输入q退出）:")
        if input_str.lower() == 'q':
            break
        output_str = generator.generate(input_str)
        print(f"生成下联: {output_str}")
