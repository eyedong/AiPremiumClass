# 使用fasttext训练文本分类模型
import fasttext

#无监督模型训练
model = fasttext.train_supervised(
    input='cooking.stackexchange.txt',
    lr=0.1,
    dim=200,
    epoch=50, 
    minCount=3,
    minCountLabel=2
    )
#print('文档词汇表：', model.words)
print('文档词汇长度：', len(model.words))

#word2vec模型使用，进行文本分类

#文本分类
print(model.predict("Baking chicken in oven, but keeping it moist"))

#保存模型
#model.save_model('./cooking_word2vec.bin')
