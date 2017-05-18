# -*- coding:utf-8 -*-
import numpy as np
from collections import defaultdict

class cart_node:  # cart节点
    def __init__(self):
        self.split_point=None
        self.f_index=None
        self.left_node=None
        self.right_node=None
        self.labels=[]
        self.label=None



def feature_min_gini(f,y):
    '''
    得到特征下的最佳分裂点和gini值
    :param f: 特征f的向量
    :param y: 标签向量
    :return: 分裂点和gini值
    '''
    d=dict()
    y_set=set()
    for f_value,y in zip(f,y):
        y_set.add(y)
        if f_value not in d.keys():
            d[f_value]={y:1}
        else:
            if y not in d[f_value].keys():
                d[f_value][y]=1
            else:
                d[f_value][y]+=1
    y_2_index=dict()
    for i,y_v in enumerate(y_set):
        y_2_index[y_v]=i
    l=list(d.keys())
    l.sort()
    split_point=[]
    for i in range(len(l)-1):
        split_point.append((l[i]+l[i+1])/2)

    data_num=len(f)
    min_gini=1
    min_point=None
    y_num=len(y_set)
    for point in split_point:
        mat=[[0 for i in range(y_num)],[0 for i in range(y_num)]]
        for f_value,t in d.items():
            if f_value<point:
                for y_value,count in t.items():
                    mat[0][y_2_index[y_value]]+=count
            else:
                for y_value,count in t.items():
                    mat[1][y_2_index[y_value]]+=count
        gini=0
        for i in range(2):
            g = 0
            num=sum(mat[i])
            for count in mat[i]:
                g+=(count/num)**2
            gini+=(1-g)*num/data_num
        if gini<=min_gini:
            min_point=point
            min_gini=gini
    return min_gini,min_point

def search(root,x):
    pos=root
    while pos.right_node and pos.left_node:

        value=x[pos.f_index]
        if value<pos.split_point:
            pos=pos.left_node
        else:
            pos=pos.right_node
    return pos.label

class Cart:
    def __init__(self):
        # self.deepth=deepth
        pass

    def creat_cart(self,X,y,f_indexs):
        #结束条件
        if len(X)==0:
            node = cart_node()
            node.label = Counter(y).most_common(1)[0][0]
            node.labels = y
            return node

        y_label = np.unique(y)

        if len(y_label) == 1:  # 树下标签一致
            node=cart_node()
            node.label=y_label[0]
            node.labels=y
            return node

        data_num, f_num = X.shape
        min_gini=1
        split_point=None
        index=None
        for f_index in range(f_num):
            f=X[:,f_index]
            gini,point=feature_min_gini(f,y)
            if gini<=min_gini:
                min_gini=gini
                split_point=point
                index=f_index
        f_index=f_indexs[index]

        root=cart_node()
        root.split_point=split_point
        root.f_index=f_index
        root.labels=y

        if f_num==1:
            x_new=[]
        X_left=[]
        X_right=[]
        y_left=[]
        y_right=[]
        for i,f_value in enumerate(X[:,index]):
            if f_value<split_point:
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                X_right.append(X[i])
                y_right.append(y[i])
        if f_num==1:
            X_left=[]
            X_right=[]
            f_nums=[]
        else:
            f_indexs = np.delete(f_indexs, index, axis=0)
            X_left=np.delete(X_left,index,axis=1)
            X_right = np.delete(X_right, index, axis=1)
        root.left_node=self.creat_cart(X_left,y_left,f_indexs)
        root.right_node=self.creat_cart(X_right,y_right,f_indexs)
        return root


    def fit(self,X,y):
        self.X=X
        self.y=y
        self.data_num,self.f_num=X.shape
        self.f_indexs=[i for i in range(self.f_num)]
        self.root=self.creat_cart(self.X,self.y,self.f_indexs)



    def predict(self,X):
        X = np.atleast_2d(X)
        for x in X:
            label = search(self.root, x)
            print(label)


if __name__ == '__main__':
    y = np.array([1, 0, 1, 0])
    X = np.array([[1, 2, 4, 0, 2], [2, 1, 4, 0, 3], [0, 3, 1, 1, 3], [1, 2, 3, 1, 2]])
    cart=Cart()
    cart.fit(X,y)

    cart.predict([2, 1, 4, 0, 3])