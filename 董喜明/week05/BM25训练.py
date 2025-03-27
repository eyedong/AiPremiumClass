# BM25训练,实现基于豆瓣top250图书评论的简单推荐系统

from tqdm import tqdm
import csv
import jieba
import numpy as np
from rank_bm25 import BM25Okapi

# 数据预处理

# 修复后文件存盘文件
fixed = open("data/Douban/doubanbook_top250_comments_BM25.txt", "w", encoding="utf-8")

# 修复前内容文件
lines = [line for line in open("data/Douban/doubanbook_top250_comments.txt", "r", encoding="utf-8")]

for i, line in enumerate(lines):
    #保存标题列
    if i == 0:
        fixed.write(line)
        prev_line = ''                      #上一行的书名置为空
        continue
    #提取书名和评论文本
    terms = line.split("\t")

    #当前行的书名 == 上一行的书名
    if terms[0] == prev_line.split("\t")[0]:
        if len(prev_line.split("\t")) == 6: #上一行是评论
            #保存上一行记录
            fixed.write(prev_line + '\n')   
            prev_line = line.strip()        #保存当前行
        else:
            prev_line = ""
    else:
        if len(terms) == 6:                 #新书评论
            prev_line = line.strip()        #保存当前行
        else:
            prev_line += line.strip()       #合并当前行和上一行
fixed.close()

def load_data(doubanbook_top250_comments, stop_words):
    #图书评论数据集合
    book_comments = {}                      #{书名：评论词+评论词......}

    with open(doubanbook_top250_comments, 'r', encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter='\t')  #识别格式文本中标题
        for item in reader:
            book = item['book']
            comment = item['body']
            comment_words = [word for word in jieba.lcut(comment) if word not in stop_words]

            if not book or not comment:      #跳过空书名
                continue

            # 图书评论数据收集
            book_comments[book] = book_comments.get(book, [])
            book_comments[book].extend(comment_words)
    return book_comments

# 使用BM25算法
def main():
    # 加载停用词表
    stop_words = [line.strip() for line in open("data/Douban/stopwords.txt", "r", encoding="utf-8")]
    
    # 调用函数加载修复后评论数据
    book_comments = load_data("doubanbook_top250_comments.txt", stop_words) 

    # 加载图书评论文本（分词后的列表）
    book_names = list(book_comments.keys())
    tokenized_corpus = list(book_comments.values())  # 每个元素是分词后的列表
    
    # 检查数据格式
    if not all(isinstance(doc, list) for doc in tokenized_corpus):
        raise ValueError("数据必须是分词后的列表")

    # 构建BM25模型
    bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)  # 参数可调整

    # 构建相似度矩阵（BM25得分）
    n = len(tokenized_corpus)
    similarity_matrix = np.zeros((n, n))
    for i in tqdm(range(n), desc="计算BM25相似度"):
        scores = bm25.get_scores(tokenized_corpus[i])
        # 归一化处理（将得分缩放到0-1之间）
        max_score = np.max(scores)
        if max_score > 0:
            scores = scores / max_score
        similarity_matrix[i] = scores

    # 输入要推荐的图书名称
    input_book = input("\n请输入图书名称：")
    try:
        book_idx = book_names.index(input_book)
    except ValueError:
        print(f"未找到图书：{input_book}")
        return

    # 获取推荐（按BM25得分排序）
    recommend_indices = np.argsort(-similarity_matrix[book_idx])[1:20]  # 排除自己
    
    # 输出结果
    print("\n推荐结果（按BM25相似度排序）：")
    for idx in recommend_indices:
        print(f"《{book_names[idx]}》\t 相似度：{similarity_matrix[book_idx][idx]:.4f}")

if __name__ == '__main__':
    main()
