# TF-IDF训练,实现基于豆瓣top250图书评论的简单推荐系统

from tqdm import tqdm
import csv
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 修复后文件存盘文件
fixed = open("data/Douban/doubanbook_top250_comments_TFIDF.txt", "w", encoding="utf-8")

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

if __name__ == '__main__':

    # 加载停用词表
    stop_words = [line.strip() for line in open("data/Douban/stopwords.txt", "r", encoding="utf-8")]
    
    # 调用函数加载修复后评论数据
    book_comments = load_data("doubanbook_top250_comments.txt", stop_words) 

    # 加载图书评论文本
    book_names = []
    book_comms = []
    for book, comments in book_comments.items():
        book_names.append(book)
        book_comms.append(comments)

    # 检查数据是否加载成功
    if not book_comments:
        raise ValueError("未加载到有效数据，请检查输入文件或逻辑")
            
    # 打印前3本书的评论词数量
    #for book, comms in list(book_comments.items())[:3]:
    #    print(f"书名：{book}，评论词数量：{len(comms)}")

    # 构建TF-IDF特征矩阵
    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        max_features=5000,                  # 增加特征维度
        ngram_range=(1, 2),                 # 包含1-2个词的组合
        min_df=2,                           # 忽略出现少于2次的词
        max_df=0.75                         # 忽略出现在75%以上文档的词
        )
    tfidf_matrix = vectorizer.fit_transform([' '.join(comms) for comms in book_comms])

    # 计算图书之间的余弦相似度
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # 输入要推荐的图书名称
    book_list = list(book_comments.keys())      #获取完整的图书名称列表
    input_book = input("\n请输入图书名称：")
   
    try:
        # 获取图书索引（使用精确匹配）
        book_idx = book_list.index(input_book)
    except ValueError:
        # 模糊匹配容错（处理书名输入不全的情况）
        matches = [idx for idx, name in enumerate(book_list) if input_book in name]
        if matches:
            book_idx = matches[0]
            print(f"找到近似匹配：《{book_list[book_idx]}》")
        else:
            print(f"未找到包含「{input_book}」的图书")
            exit()

    # 修改后的推荐结果显示逻辑
    print(f"\n基于《{book_list[book_idx]}》的推荐结果：")
    recommend_book_index = np.argsort(-similarity_matrix[book_idx])[1:20]  # 排除自己

    # 获取相似度分数并过滤无效值
    shown = 0
    for idx in recommend_book_index:
        similarity = similarity_matrix[book_idx][idx]
        # 显示条件调整：相似度>0.01且最多显示20条
        if similarity > 0.01 and shown < 20:
            print(f"《{book_list[idx]}》\t 相似度：{similarity:.4f}")
            shown += 1
    
    if shown == 0:
        print("（无有效推荐，可能原因：")
        print("1. 该图书评论数据不足")
        print("2. 系统未找到有效关联图书")
        print("3. 请尝试其他图书查询）")
