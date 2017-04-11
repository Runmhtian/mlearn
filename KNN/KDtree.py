# -*- coding:utf-8 -*-
import numpy as np
from numpy.linalg import norm
from collections import Counter
MAX=9999


class KD_Node:
    def __init__(self, vector=None, split=None, left=None, right=None):
        self.vector = vector
        self.split = split
        self.left = left
        self.right = right
        self.category=None  # 标签



def create_kdtree(X,y):
    if type(X)==np.ndarray:
        X.tolist()
    if len(X) == 0:
        return
    split = np.argmax(np.var(X, axis=0))  # 获取方差最大的列标
    data_list = X.tolist()
    # X与y一一对应，一起排序
    for x,category in zip(data_list,y):
        x.append(category)
    data_list.sort(key=lambda x: x[split])  # 按照split进行排序
    y=[]
    for x in data_list:
        y.append(x.pop())
    split_vector_index=int(len(data_list) / 2)
    split_vector = data_list[split_vector_index] # 数据分割
    root = KD_Node(split_vector, split)
    root.category=y[split_vector_index]
    X=np.array(data_list)
    root.left = create_kdtree(X[0:split_vector_index],y[0:split_vector_index])
    root.right = create_kdtree(X[split_vector_index + 1:],y[split_vector_index + 1:])
    return root

def search(trace_list,pos,data):
    while pos:
        trace_list.append(pos)
        split = pos.split
        if data[split] > pos.vector[split]:
            pos = pos.right
        else:
            pos = pos.left

def nearest(root,data):
    if type(data)==list:
        data=np.array(data)
    pos=root
    trace_list=[]
    search(trace_list,root,data)
    nearest_node=None
    min_dist=MAX
    while len(trace_list)!=0:
        node=trace_list.pop()
        # print(node.vector)
        if node.left==None and node.right==None:  # 若是叶子节点
            dist=norm(np.array(node.vector)-data)
            nearest_node=node
            min_dist=dist
        else:
            dist = norm(np.array(node.vector) - data)
            if dist<=min_dist:  # 向trace_list 添加另一子树
                nearest_node = node
                min_dist=dist
                if data[node.split]<node.vector[node.split]:
                    pos=node.right
                else:
                    pos=node.left
                search(trace_list,pos,data)
    return nearest_node



def Knearest(k,root,data):
    if type(data)==list:
        data=np.array(data)
    pos=root
    trace_list=[]   # 记录路径
    k_node=[]  # 记录最终结果
    k_dist=[]  # 记录距离
    search(trace_list,root,data)
    nearest_node=None
    while len(trace_list)!=0:
        node=trace_list.pop()
        # print(node.vector)
        if node.left==None and node.right==None:  # 若是叶子节点
            dist=norm(np.array(node.vector)-data)
            if len(k_node)<k:
                k_node.append(node)
                k_dist.append(dist)
            else:
                if dist<max(k_dist):
                    index=k_dist.index(max(k_dist))
                    k_dist[index]=dist
                    k_node[index]=node
        else:
            dist = norm(np.array(node.vector) - data)
            if len(k_node)<k:  # 若k_node没有满，直接加入并且另一个子树需要寻找
                k_node.append(node)
                k_dist.append(dist)
                pos = node.right if data[node.split] < node.vector[node.split] else node.left
                search(trace_list, pos, data)
            elif len(k_node)==k:
                if dist<max(k_dist):  # k_node已满，小于最大距离加入并且另一个子树需要寻找
                    index=k_dist.index(max(k_dist))
                    k_dist[index]=dist
                    k_node[index]=node
                    pos=node.right if data[node.split]<node.vector[node.split] else node.left
                    search(trace_list,pos,data)
    return k_node




if __name__ == '__main__':
    #{(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)}
    X=np.array([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]])
    y=[1,0,1,0,1,0]
    root=create_kdtree(X,y)
    for i in Knearest(3,root,[2,4.5]):
        print(i.vector,i.category)
