# -*- coding:utf-8 -*-

import heapq
import numpy as np
from collections import Counter
from numpy.linalg import norm

import KDtree


def hamming_distance(s):  # s1-s2=s
    sum=0
    for i in s:
        if i!=0:
            sum+=1
    return sum


class KNN_Model(object):
    '''
    algorithm:
    burte  暴力算法  挨个计算距离
    kd-tree  使用kd-tree数据结构
    
    metric：
    euclidean  欧氏距离
    hamming  海明距离
    '''
    def __init__(self, k, algorithm='brute',metric='euclidean'):
        self.k = k
        self.algorithm = algorithm
        if metric=='euclidean':
            self.metric = norm
        elif metric=='hamming':
            self.metric=hamming_distance

    def fit(self, X, y):
        self.X = X
        self.y = y
        if self.algorithm == 'brute':
            return
        elif self.algorithm == 'kd-tree':
            self.kd_tree_root= KDtree.create_kdtree(X, y)


    def predict(self, X_test):
        pred = []
        if self.algorithm=='brute':
            for x in X_test:
                distance = [self.metric(x - x_train) for x_train in X]  # 计算距离
                k_index = heapq.nsmallest(k, range(len(distance)), distance.__getitem__)  # 获取距离最小的前k个距离的下标
                l = Counter(self.y[k_index]).most_common()  # 根据下标获取y中的类别，并统计频次，排序
                pred.append(l[0][0])
        if self.algorithm=='kd-tree':
            if self.metric==hamming_distance:
                raise RuntimeError('kd-tree 不适合用海明距离')
            for x_test in X_test:
                k_node = KDtree.Knearest(self.k, self.kd_tree_root, x_test,self.metric)
                print('%s k nearest %s'%(str(x_test),str([node.vector for node in k_node])))
                l = Counter([node.category for node in k_node]).most_common()
                pred.append(l[0][0])
        return pred

if __name__ == '__main__':
    X = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    y = np.array([1, 0, 1, 0, 1, 0])
    X_test=np.array([[2,3],[9,5]])
    knn=KNN_Model(2,algorithm='kd-tree')
    knn.fit(X,y)
    print(knn.predict(X_test))