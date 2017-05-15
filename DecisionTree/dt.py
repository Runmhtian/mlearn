# -*- coding:utf-8 -*-
from collections import Counter,defaultdict
from math import log
import numpy as np

class DT_node:
    def __init__(self,f_index):
        self.f_index=f_index
        self.sub_node_d=dict()

class DT_label_node:
    def __init__(self,label):
        self.label=label

def cal_entropy(y):
    y=np.atleast_1d(y)
    entropy=0
    y_values,y_counts=np.unique(y,return_counts=True)
    n=y.shape[0]
    for y_count in y_counts:
        entropy+=(y_count/n)*log((y_count/n),2)
    return -entropy


def find_array(narray,l):
    templist=[]
    for i in l:
        templist.append(narray[i])
    return np.array(templist)
# ID3

class DecisionTree:
    def __init__(self,algo='id3'):
        self.algo='id3'

    def creat_tree(self,X,y):
        # print('-------')
        if len(X)==0:  # 特征向量为空
            return DT_label_node(label=Counter(y).most_common(1)[0][0])
        y_label=np.unique(y)

        if len(y_label)==1:  # 树下标签一致
            return DT_label_node(label=y_label[0])
        n, f_num = X.shape

        flag=True  # 特征值都一样，提前结束
        for x in X:
            l=np.equal(x,X[0])
            if False in l:
                flag=False
                break
        if flag:
            return DT_label_node(label=Counter(y).most_common(1)[0][0])
        y_entropy=cal_entropy(y)
        max_entropy=0
        index=None
        max_f_value_2_list=None
        # print(X)
        # print(y)
        for i in range(f_num):
            # 特征值离散
            f_values,f_indexs,f_counts=np.unique(X[:,i],return_inverse=True,return_counts=True)
            # print(f_values,f_indexs)
            f_value_2_p=dict()
            en=0
            for f_value,f_count in zip(f_values,f_counts):
                f_value_2_p[f_value]=f_count/n
            f_value_2_list=defaultdict(list)
            for j,f_index in enumerate(f_indexs):
                f_value_2_list[f_values[f_index]].append(j)
            # print(i)
            # print(f_value_2_list)
            for f_value,f_list in f_value_2_list.items():
                templist=[]
                for j in f_list:
                    templist.append(y[j])
                en+=f_value_2_p[f_value]*cal_entropy(templist)
            # print(en)
            if en>=max_entropy:
                index=i
                max_entropy=en
                max_f_value_2_list=f_value_2_list
        root=DT_node(f_index=index)
        # print(index)
        # print(max_f_value_2_list)
        for f_value, f_list in max_f_value_2_list.items():
            # print(f_value)
            # print(find_array(X,f_list))
            if f_num==1:
                X_new=[]
            else:
                X_new=np.delete(find_array(X,f_list),index,axis=1)
            y_new=find_array(y,f_list)
            root.sub_node_d[f_value]=self.creat_tree(X_new,y_new)
        return root

    def fit(self,X,y):
        self.X=X
        self.y=y
        self.root=self.creat_tree(X,y)

    def predict(self,X):
        pass

if __name__ == '__main__':
    y = np.array([1, 0, 1, 0])
    X = np.array([[1, 2, 4, 0, 2], [2, 1, 4, 0, 3], [0, 3, 1, 1, 3], [1, 2, 3, 1, 2]])
    dt=DecisionTree()
    root=dt.creat_tree(X,y)
    print(root.f_index)
    for i,j in root.sub_node_d.items():
        print(i,j)