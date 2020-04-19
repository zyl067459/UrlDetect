# coding: utf-8

import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
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
    with open('model/lgs.pickle', 'rb') as input:
        w = pickle.load(input)
    with open('test.txt', 'r') as f1:
        for i in range(0, 2000):
            result1.append(f1.readline().strip())
    # print(result1.__len__())
    y_true = []
    for i in range (0,1000):
        y_true.append(1)
    for i in range (1000,2000):
        y_true.append(0)
    y_pred = []
    w.predict(result1)
    y_pred = w.predicty(result1)

    print(classification_report(y_true, y_pred))
