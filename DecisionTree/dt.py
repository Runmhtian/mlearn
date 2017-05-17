# -*- coding:utf-8 -*-
from collections import Counter,defaultdict
from math import log
import numpy as np
'''
实现一个简易版的决策树
'''
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

    def creat_tree(self,X,y,f_index_l):

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

        for i in range(f_num):
            # 特征值离散
            f_values,f_indexs,f_counts=np.unique(X[:,i],return_inverse=True,return_counts=True)

            f_value_2_p=dict()
            en=0
            for f_value,f_count in zip(f_values,f_counts):
                f_value_2_p[f_value]=f_count/n
            f_value_2_list=defaultdict(list)
            for j,f_index in enumerate(f_indexs):
                f_value_2_list[f_values[f_index]].append(j)

            for f_value,f_list in f_value_2_list.items():
                templist=[]
                for j in f_list:
                    templist.append(y[j])
                en+=f_value_2_p[f_value]*cal_entropy(templist)
            # print(en)
            if en>=max_entropy:
                index=i
                f_index=f_index_l[index]
                max_entropy=en
                max_f_value_2_list=f_value_2_list
        root=DT_node(f_index=f_index)

        for f_value, f_list in max_f_value_2_list.items():

            if f_num==1:
                X_new=[]
                f_index_l=[]
            else:
                X_new=np.delete(find_array(X,f_list),index,axis=1)
                f_new_index_l=[]
                for i in range(len(f_index_l)):
                    if i!=index:
                        f_new_index_l.append(f_index_l[i])
            y_new=find_array(y,f_list)
            # print(y_new)
            root.sub_node_d[f_value]=self.creat_tree(X_new,y_new,f_new_index_l)
        return root

    def fit(self,X,y):
        self.X=X
        self.y=y
        self.f_index_l=[i for i in range(X.shape[1])]
        self.root=self.creat_tree(X,y,self.f_index_l)

    def search(self,root,x):
        pos=root
        while not isinstance(pos,DT_label_node):
            value=x[pos.f_index]
            pos=pos.sub_node_d.get(value,None)
            if pos==None:
                raise RuntimeError('%s,%d 在训练集中不存在'%(str(x),f_index))
        return pos.label


    def predict(self,X):
        X=np.atleast_2d(X)
        for x in X:
            label=self.search(self.root,x)
            print(label)


if __name__ == '__main__':
    y = np.array([1, 0, 1, 0])
    X = np.array([[1, 2, 4, 0, 2], [2, 1, 4, 0, 3], [0, 3, 1, 1, 3], [1, 2, 3, 1, 2]])
    dt=DecisionTree()
    dt.fit(X,y)
    # print(dt.root.f_index)
    # for i,j in dt.root.sub_node_d.items():
    #     print(i,j)
    # node=dt.root.sub_node_d[3]
    # print(node.f_index)
    # for i,j in node.sub_node_d.items():
    #     print(i,j)
    dt.predict([2,2,4,0,3])