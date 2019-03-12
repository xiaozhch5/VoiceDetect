# -*- coding:UTF-8 -*-

from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import operator
from functools import reduce


trainset_num = 100  # 各自取50

result_tamper = scio.loadmat('result_tamper.mat')  # 读入字典
result_original = scio.loadmat('result_original.mat')
tamper_result = []
original_result = []
result = []
rresult = np.zeros((trainset_num, 5400))
testset = np.zeros((100, 5400))
for i in range(100):
    tamper_result.append(result_tamper['result_tamper'][0][0])
    original_result.append(result_original['result_original'][0][0])
tamper_result_array = np.array(tamper_result)  # shape为（100， 3， 1800）
original_result_array = np.array(original_result)  # shape为（100， 3， 1800）
result = np.concatenate((tamper_result_array[0:50], original_result_array[0:50]), axis=0)  # 各自取50个作为训练集
for j in range(trainset_num):
    rresult[j] = result[j].flatten()
for k in range(50):
    testset[k] = tamper_result_array[k + 50].flatten()

for l in range(50, 100):
	testset[l] = original_result_array[l].flatten()

label = np.array([1] * 50 + [0] * 50)  # 标签


svc = SVC(kernel='poly', degree=2, gamma=1, coef0=0)
svc.fit(rresult, label)
np.random.shuffle(testset)
pre = svc.predict(testset)
print(sum(pre))