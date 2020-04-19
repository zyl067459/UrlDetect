# coding: utf-8

import os

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import urllib
import time
import pickle
import html


class WAF(object):
    def __init__(self):
        good_query_list=[]
        bad_query_list=[]
        print("开始读取url......")
        good_query = open('goodqueries.txt', 'r', encoding='UTF-8').readlines()
        for i in good_query:
            i = str(urllib.parse.unquote(i))  # 对url解码
            good_query_list.append(i)
        good_query_list=list(set(good_query_list))
        bad_query = open('badqueries.txt', 'r', encoding='UTF-8').readlines()
        for i in bad_query:
            i = str(urllib.parse.unquote(i))  # 对url解码
            bad_query_list.append(i)
        bad_query_list=list(set(bad_query_list))

        good_y = []
        bad_y = []
        for i in range(0, len(good_query_list)):
            good_y.append(0)
        for i in range(0, len(bad_query_list)):
            bad_y.append(1)

        queries = bad_query_list + good_query_list
        y = bad_y + good_y
        print("开始处理url......")
        # 转化为tf-idf的特征矩阵25
        self.vectorizer = TfidfVectorizer(tokenizer=self.get_ngrams)
        # 使用fit_transform学习的词汇和文档频率（df），将文档转换为文档 - 词矩阵。返回稀疏矩阵
        X = self.vectorizer.fit_transform(queries)
        # print(X)······················
        # 使用 train_test_split 分割 X y 列表
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0)
        print("模型正在训练......")
        # 定理逻辑回归方法模型
        self.lgs = LogisticRegression()

        # 使用逻辑回归方法训练模型实例 lgs
        self.lgs.fit(X_train, y_train)

        # 使用测试值 对 模型的准确度进行计算
        print('模型的准确度:{}'.format(self.lgs.score(X_test, y_test)))


    # 对 新的请求列表进行预测
    def predict(self, new_queries):
        new_queries = [urllib.parse.unquote(url) for url in new_queries]
        X_predict = self.vectorizer.transform(new_queries)
        res = self.lgs.predict(X_predict)
        res_list = []
        for q, r in zip(new_queries, res):
            if r==0:
                tmp = '正常请求'
            else:
                tmp = '恶意请求'
            q = html.escape(q)
            res_list.append({'url': q, 'res': tmp})
        for n in res_list:
            print(n)
        return res_list

    def predicty(self, new_queries):
        new_queries = [urllib.parse.unquote(url) for url in new_queries]
        X_predict = self.vectorizer.transform(new_queries)
        res = self.lgs.predict(X_predict)
        return res

    # 分割url
    def get_ngrams(self, query):
        tempQuery = str(query)
        ngrams = []
        for i in range(0, len(tempQuery) - 3):
            ngrams.append(tempQuery[i:i + 3])
        return ngrams


if __name__ == '__main__':
    # 若 检测模型文件lgs.pickle 不存在,需要先训练出模型


    w = WAF()

    with open('model/lgs.pickle', 'wb') as output:
         pickle.dump(w,output)
    print("模型训练完成")


    # result1 = []
    # result2 = []
    # with open('lgs.pickle', 'rb') as input:
    #     w = pickle.load(input)
    # with open('good_test.txt', 'r') as f1:
    #     for i in range(0, 10):
    #         result1.append(f1.readline().strip())
    #
    # with open('bad_test.txt', 'r', encoding='utf-8') as f2:
    #     for i in range(0, 10):
    #         result2.append(f2.readline().strip())
    #
    # result = result2 + result1
    #
    # w.predict(result)


