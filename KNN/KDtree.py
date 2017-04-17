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
    # print(np.var(X, axis=0))
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
    # print(split_vector)
    root = KD_Node(split_vector, split)
    root.category=y[split_vector_index]
    X=np.array(data_list)
    root.left = create_kdtree(X[:split_vector_index],y[:split_vector_index])
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

# def nearest(root,data):
#     if type(data)==list:
#         data=np.array(data)
#     pos=root
#     trace_list=[]
#     search(trace_list,root,data)
#     nearest_node=None
#     min_dist=MAX
#     while len(trace_list)!=0:
#         node=trace_list.pop()
#         # print(node.vector)
#         if node.left==None and node.right==None:  # 若是叶子节点
#             dist=norm(np.array(node.vector)-data)
#             nearest_node=node
#             min_dist=dist
#         else:
#             dist = norm(np.array(node.vector) - data)
#             if abs(data[node.split]-node.vector[node.split])<=min_dist:  # 向trace_list 添加另一子树
#                 nearest_node = node
#                 min_dist=dist
#                 if data[node.split]<node.vector[node.split]:
#                     pos=node.right
#                 else:
#                     pos=node.left
#                 search(trace_list,pos,data)
#     return nearest_node


'''
kd-tree  k最近邻搜索
给定一个构建于一个样本集的 kd 树，寻找距离点 p 最近的 k 个样本。

设 L为一个有 k 个空位的列表，用于保存已搜寻到的最近点。
步骤1  根据 p 的坐标值和每个节点的切分向下搜索，记录寻找的路径到栈X中。
步骤2  弹出栈一个节点，直到栈为空，返回L。
   》若是叶子节点，如果 L 里不足 k 个点，则将当前节点坐标加入 L ；如果 L 已有k个点，计算p与当前点的距离，
          若小于 L 中离 p 最远的点则替换。	
   》若不是叶子节点：
	如果 L 里不足 k 个点，则将当前节点坐标加入 L ，并在当前节点的另一子树上执行步骤1;
	如果 L 已有k个点，计算p与当前点的距离，小于 L 中离 p 最远的点则替换。计算 p 和当前节点切分线的距离若该距离小于 L 中
	距离 p 最远的距离，在当前节点的另一子树上执行步骤1

'''
def Knearest(k,root,data,metric):  # 欧氏距离
    if type(data)==list:
        data=np.array(data)
    trace_list=[]   # 记录路径
    k_node=[]  # 记录最终结果
    k_dist=[]  # 记录距离
    search(trace_list,root,data)
    while len(trace_list)!=0:
        node=trace_list.pop()
        print(node.vector)
        if node.left==None and node.right==None:  # 若是叶子节点
            dist=metric(np.array(node.vector)-data)
            if len(k_node)<k:
                k_node.append(node)
                k_dist.append(dist)
            else:
                if dist<max(k_dist):
                    index=k_dist.index(max(k_dist))
                    k_dist[index]=dist
                    k_node[index]=node
        else:
            dist = metric(np.array(node.vector) - data)
            if len(k_node)<k:  # 若k_node没有满，直接加入并且另一个子树需要寻找
                k_node.append(node)
                k_dist.append(dist)
                pos = node.right if data[node.split] < node.vector[node.split] else node.left
                search(trace_list, pos, data)
            elif len(k_node)==k:
                if dist<max(k_dist):
                    index = k_dist.index(max(k_dist))
                    k_dist[index] = dist
                    k_node[index] = node
                # k_node已满，p到当前节点切分线的距离，小于k_dist中的最大距离，则另一子树需要寻找
                if abs(data[node.split]-node.vector[node.split])<max(k_dist):
                    pos=node.right if data[node.split]<node.vector[node.split] else node.left
                    search(trace_list,pos,data)
    return k_node








