# -*- coding:utf-8 -*-

import numpy as np
from collections import defaultdict
import math

def _cal_p_y(y):  # 计算概率
    y_2_p=dict()
    y_indexs,y_counts=np.unique(y,return_counts=True)
    n=y.shape[0]
    for y_index,y_count in zip(y_indexs,y_counts):
        y_2_p[y_index]=(y_count+1)/(n+1)
    return y_2_p


def _cal_p_x(y):  # 计算概率
    y_2_p=dict()
    y_indexs,y_counts=np.unique(y,return_counts=True)
    n=y.shape[0]
    for y_index,y_count in zip(y_indexs,y_counts):
        y_2_p[y_index]=(y_count+1)/(n+1)  # 都加1
    y_2_p['null'] = 1 / (n + 1)  # 其他值概率
    return y_2_p


# 离散特征处理
def _cal_dispersed(f,y):  # 计算单个特征的条件概率
    # print(f)
    d=defaultdict(list)
    ret_d=dict()
    y_s,y_index=np.unique(y,return_inverse=True)  # y_index 对应位置的y值
    # print(y_s)
    # print(y_index)
    for i,index in enumerate(y_index):
        d[y_s[index]].append(i)
    for y_value,x_index_list in d.items():
        temp_list=[]
        for i in x_index_list:
            temp_list.append(f[i])
        ret_d[y_value]=_cal_p_x(np.array(temp_list))
    return ret_d



# 连续特征处理
def cal_gaussion(f,y):
    d = defaultdict(list)
    ret_d = dict()
    y_s, y_index = np.unique(y, return_inverse=True)  # y_index 对应位置的y值
    for i, index in enumerate(y_index):
        d[y_s[index]].append(i)
    for y_value,x_index_list in d.items():
        temp_list=[]
        for i in x_index_list:
            temp_list.append(f[i])
        # print(temp_list)
        avg=np.mean(temp_list)
        var=np.var(temp_list)
        # print(avg,var)
        ret_d[y_value]=(avg,var)
    return ret_d

class Bayes:

    def __init__(self,type='dispersed'):
        '''
        dispersed:离散特征
        gaussion:连续特征
        hybid：混合
        :param type: 
        '''
        self.type=type

    def _cal_p_xi_y(self,X,y):  # 计算所有特征的条件概率
        n, f_num = X.shape
        xy_2_p = dict()
        if self.type=='dispersed':
            for i in range(f_num):
                xy_2_p[i] = _cal_dispersed(X[:, i], y)
            return xy_2_p
        if self.type=='gaussion':
            for i in range(f_num):
                xy_2_p[i]=cal_gaussion(X[:, i], y)
            return xy_2_p

    def cal_pred(self,d_xy, d_y, x):
        d = dict()
        for y_value in d_y.keys():
            p_yi = d_y[y_value]
            p = p_yi
            if self.type=='dispersed':
                for i, value in enumerate(x):
                    x_2_p = d_xy[i][y_value]
                    # print(x_2_p)
                    if value in x_2_p.keys():
                        p = p * x_2_p[value]
                    else:
                        # print(i,value)
                        # 给出的样本特征值，在训练集中对应的条件概率下不存在
                        p = p * x_2_p['null']
                d[y_value] = p
            if self.type=='gaussion':
                for i, value in enumerate(x):
                    # print(d_xy[i][y_value])
                    avg,var = d_xy[i][y_value]
                    if var==0:
                        raise RuntimeError('不适合用gaussion')
                    p=p*(1/(math.sqrt(2*math.pi*var)))*math.exp(-(value-avg)**2/(2*var))
                d[y_value]=p
        return d


    def fit(self,X,y):
        self.X = X
        self.y = y
        self.d_xy=self._cal_p_xi_y(X,y)  #　条件概率估计
        self.d_y=_cal_p_y(y)  # y的概率估计
        # print(self.d_xy)

    def predict(self,X_test):
        X_test=np.atleast_2d(X_test)
        result=[]
        for x in X_test:
            # print(cal_pred(self.d_xy,self.d_y,x))
            result.append(self.cal_pred(self.d_xy,self.d_y,x))
        return result

if __name__ == '__main__':
    y=np.array([1,0,1,0])
    X=np.array([[1,2,4,0,2],[2,1,4,0,3],[0,3,1,1,3],[1,2,3,1,2]])
    # print(_cal_p_y(y))
    # print(_cal(X[:,2],y))
    # print(_cal_p_xi_y(X,y))
    b=Bayes(type='gaussion')
    b.fit(X,y)
    print(b.predict([1,3,4,1,2]))