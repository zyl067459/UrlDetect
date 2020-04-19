# coding: utf-8

import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import urllib
import time
import pickle
import html
from train import WAF


if __name__ == '__main__':
    # 若 检测模型文件lgs.pickle 不存在,需要先训练出模型
    # w = WAF()
    # with open('lgs.pickle','wb') as output:
    #      pickle.dump(w,output)


    result1 = []
    result2 = []
    with open('model/lgs.pickle', 'rb') as input:
        w = pickle.load(input)
    with open('good_test.txt', 'r') as f1:
        for i in range(0, 10):
            result1.append(f1.readline().strip())

    with open('bad_test.txt', 'r', encoding='utf-8') as f2:
        for i in range(0, 10):
            result2.append(f2.readline().strip())

    result = result2 + result1
    y_pred = []
    w.predict(result)
    # y_pred = w.predicty(result)
    # print(y_pred)



