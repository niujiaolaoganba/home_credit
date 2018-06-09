# /usr/bin/env python3
# -*-coding:utf-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from task import TaskOptimizer



# 准备好数据
train = pd.read_csv('../data/input/train.csv')

X_train = train.drop('TARGET', axis = 1)
y_train = train.TARGET
X_test= pd.read_csv('../data/input/test.csv')
# y_test = test.is_reg
X_train.columns = [ 'f_' + str(i) for i in range(X_train.shape[1])]
X_test.columns = [ 'f_' + str(i) for i in range(X_train.shape[1])]
print(X_train.shape)
print(X_train.columns)

#跑任务
optimizer = TaskOptimizer(X_train, y_train, X_test, y_test = None, cv = 3, max_evals = 10, verbose=True)
optimizer.run()


