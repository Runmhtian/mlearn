# -*- coding:utf-8 -*-
'''
使用KNN算法来实现手写数字的识别
特征向量 32*32的01矩阵  也就是1*1024向量
'''
import os
import numpy as np
import KNN
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def file2vector(filepath):
    vector=[]
    with open(filepath,'r') as f:
        for line in f:
            vector.extend([int(s) for s in line.strip()])
    assert len(vector)==1024
    return vector


def vector_label(path):
    X=[]
    y=[]
    filenames=[]
    for file in os.listdir(path):
        filenames.append(os.path.join(path,file))
        label=int(file.split('_')[0])
        y.append(label)
        X.append(file2vector(os.path.join(path,file)))
    return np.array(X),np.array(y),filenames

def evaluate_acc(y_true,y_pred):
    correct_num = 0
    for i,j in zip(y_true,y_pred):
        if i==j:
            correct_num+=1
    acc=correct_num/y_true.shape[0]
    return acc

if __name__ == '__main__':
    trainpath=os.getcwd()+r'\hand_digit\trainingDigits'
    testpath=os.getcwd()+r'\hand_digit\testDigits_110'
    X,y,_=vector_label(trainpath)
    X_test,y_test,testfilenames=vector_label(testpath)
    print('训练集 %d个'%X.shape[0])
    print('测试集 %d个'%X_test.shape[0])

    # knn=KNN.KNN_Model(5)
    # knn.fit(X,y)
    # rand_indexs=random.sample([i for i in range(110)],10)
    # for i in rand_indexs:
    #     p=knn.predict(np.array([X_test[i]]))
    #     print('文件%s 被预测为%d'%(testfilenames[i],p[0]))

    # # 自己实现的

    for k in [3,5,10]:
        myknn=KNN.KNN_Model(k,metric='euclidean')
        myknn.fit(X,y)
        pred=myknn.predict(X_test)
        print('k=%d 预测正确率为 %f'%(k,evaluate_acc(y_test,pred)))
    print()
    # sklearn
    knn=KNeighborsClassifier(algorithm='brute')
    # 参数选择
    param_grid = {'n_neighbors': [i for i in range(3, 10)]}
    grid = GridSearchCV(knn, param_grid,n_jobs=3)
    grid.fit(X, y)
    print("The best parameters are %s"%grid.best_params_)
    model = grid.best_estimator_
    y_pred=model.predict(X_test)
    print('acc=%f'%accuracy_score(y_test,y_pred))

    # 其他基本分类器  使用默认参数
    print()
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB,MultinomialNB
    from sklearn.tree import DecisionTreeClassifier
    knn=KNeighborsClassifier()
    rf = RandomForestClassifier()
    bayes1 = MultinomialNB()
    bayes2=GaussianNB()
    dt = DecisionTreeClassifier()
    for classifier in [knn,rf,bayes1,bayes2,dt]:
        classifier.fit(X,y)
        y_pred=classifier.predict(X_test)
        print(str(classifier).split('(')[0]+': %f'%accuracy_score(y_test,y_pred))




