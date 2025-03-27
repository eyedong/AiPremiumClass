# fasttext训练词向量模型，并计算词汇间的相关度
import fasttext
import jieba

# 自定义文档分词预处理
#with open("data/Douban/fengshenyanyi.txt", "r", encoding="gbk") as f:
#    lines = f.read()
#
#with open("data/Douban/fengshenyanyi_skipgram.txt", "w", encoding="utf-8") as f:
#    f.write(' '.join(jieba.cut(lines)))

#无监督模型训练
model = fasttext.train_unsupervised(
    'fengshenyanyi.txt', 
    model='skipgram', 
    lr=0.05, 
    neg=3, 
    epoch=50, 
    minCount=3
    )
#print('文档词汇表：', model.words)
print('文档词汇长度：', len(model.words))

#获取词向量
#print(model.get_word_vector('哪吒'))

#获取紧邻词
print(model.get_nearest_neighbors('哪吒', k=5))

#分析词间类比
print(model.get_analogies('哪吒', '金吒', '木吒'))

#保存模型
#model.save_model('model/fengshenyanyi_skipgram.bin')

#加载模型
#model = fasttext.load_model('./model/fengshenyanyi_skipgram.bin')

