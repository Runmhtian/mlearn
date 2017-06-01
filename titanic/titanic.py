# -*- coding:utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

train_len=len(train)
test_len=len(test)
print '训练样本%d个，测试样本%d个'%(train_len,test_len)

train_surv=train[train['Survived']==1]
train_nosurv=train[train['Survived']==0]

print '训练样本surv %d(%.1f percent)个,nosurv %d(%.1f percent)个'%\
      (len(train_surv),len(train_surv)*1./train_len*100,
       len(train_nosurv),len(train_nosurv)*1./train_len*100)


# print train.columns
'''
Index([u'PassengerId', u'Survived', u'Pclass', u'Name', u'Sex', u'Age',
       u'SibSp', u'Parch', u'Ticket', u'Fare', u'Cabin', u'Embarked'],
      dtype='object')
'''

data=pd.concat([train.drop('Survived',axis=1),test],axis=0)
# print data.describe()


def figure_show(train):
    plt.figure(figsize=(12,10))
    plt.subplot(331)
    sns.barplot('Pclass','Survived',data=train)
    plt.subplot(332)
    sns.barplot('Sex','Survived',data=train)
    plt.subplot(333)
    sns.distplot(train_surv['Age'].dropna().values,bins=range(0,81,1),color='blue',kde=False)
    sns.distplot(train_nosurv['Age'].dropna().values,bins=range(0,81,1),color='red',kde=False,axlabel='age')
    plt.subplot(334)
    sns.barplot('SibSp','Survived',data=train)
    plt.subplot(335)
    sns.barplot('Parch','Survived',data=train)
    plt.subplot(336)
    sns.barplot('Embarked','Survived',data=train)
    plt.subplot(337)
    sns.distplot(np.log10(train_surv['Fare'].dropna().values+1),kde=False,color='blue')
    sns.distplot(np.log10(train_nosurv['Fare'].dropna().values+1),kde=False,color='red',axlabel='Fare')
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.show()

# figure_show(train)


# print train.drop('PassengerId',axis=1).corr()
# sns.heatmap(train.drop('PassengerId',axis=1).corr(),vmax=0.5,annot=True)
# plt.show()

# print train.isnull().sum()
'''
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
'''

# print test.isnull().sum()
'''
PassengerId      0
Pclass           0
Name             0
Sex              0
Age             86
SibSp            0
Parch            0
Ticket           0
Fare             1
Cabin          327
Embarked         0
'''

# print train[train['Embarked'].isnull()]
'''
    PassengerId  Survived  Pclass                                       Name  \
61            62         1       1                        Icard, Miss. Amelie   
829          830         1       1  Stone, Mrs. George Nelson (Martha Evelyn)   

        Sex   Age  SibSp  Parch  Ticket  Fare Cabin Embarked  
61   female  38.0      0      0  113572  80.0   B28      NaN  
829  female  62.0      0      0  113572  80.0   B28      NaN  
'''
# tab=pd.crosstab(train['Embarked'],train['Pclass'])
# print train[(train['Pclass']==1)&(train['Sex']=='female')]
# print train[(train['Pclass']==1)&(train['Sex']=='female')]['Embarked'].value_counts()
'''
S    48
C    43
Q     1
'''
train['Embarked'].iloc[61]='S'
train['Embarked'].iloc[829]='S'

# print test[test['Fare'].isnull()]
'''
     PassengerId  Pclass                Name   Sex   Age  SibSp  Parch Ticket  \
152         1044       3  Storey, Mr. Thomas  male  60.5      0      0   3701   

     Fare Cabin Embarked  
152   NaN   NaN        S  
'''

# Fare 与 Pclass相关度最大

test['Fare'].iloc[152]=data['Fare'][data['Pclass']==3].dropna().mean().round(2)
# print train['Fare'].iloc[152]


# 定义新的特征
train['share_ticket']=0
grouped=train.groupby('Ticket')
# print grouped.get_group('SO/C 14885')
for i in range(train_len):
    train['share_ticket'].iloc[i]=len(grouped.get_group(train['Ticket'].iloc[i]))
    # print train['share_ticket'].iloc[i]

train['known_age']=train['Age'].isnull()==False
train['known_cabin']=train['Cabin'].isnull()==False
train['child']=train['Age']<10

test['share_ticket']=0
grouped=test.groupby('Ticket')
for i in range(test_len):
    test['share_ticket'].iloc[i]=len(grouped.get_group(test['Ticket'].iloc[i]))

test['known_age']=test['Age'].isnull()==False
test['known_cabin']=test['Cabin'].isnull()==False
test['child']=test['Age']<10

from sklearn.preprocessing import LabelBinarizer,OneHotEncoder
from sklearn_pandas import DataFrameMapper

# 特征处理
mapper=DataFrameMapper([
    ('known_age',LabelBinarizer()),
    ('known_cabin',LabelBinarizer()),
    ('child',LabelBinarizer()),
    ('Sex',LabelBinarizer()),
    (['Pclass'],OneHotEncoder())
],default=None)

# 选择用于训练和测试的特征
feature=['known_age','Pclass','Fare','known_cabin','child','Sex','SibSp','Parch','share_ticket']
training=train.loc[:,feature]
# print training.head(8)
X=mapper.fit_transform(training)
y=train['Survived']
testing=test.loc[:,feature]
X_test=mapper.transform(testing)
# print X
# print y
# print X_test

# from sklearn.linear_model import LogisticRegression
#
# lg=LogisticRegression()
# lg.fit(X,y)
# print lg.score(X,y)

rf_params = {
    'n_estimators': 500,
     'warm_start': True,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
}

# Extra Trees Parameters
et_params = {
    'n_estimators':500,
    'max_depth': 8,
    'min_samples_leaf': 2,
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
         'max_depth': 5,
    'min_samples_leaf': 2,
}

# Support Vector Classifier parameters
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }

from sklearn.model_selection import KFold
NFOLD=5
kf=KFold(NFOLD)
SEED=0

class SklearnHelper(object):
    def __init__(self,clf,seed=0,params=None):
        params['random_state']=seed
        self.clf=clf(**params)
    def fit(self,X,y):
        return self.clf.fit(X,y)
    def predict(self,X_test):
        return self.clf.predict(X_test)

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn import svm

rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=svm.SVC, seed=SEED, params=svc_params)

'''
OUT OF PREDICTION
oop
将多个分类器的预测结果作为训练集和测试集
列为每个分类器预测结果   训练集用NFOLD个分类器产生一列，测试集NFOLD次结果的平均值/中位数产生一列
行为每个训练样本或者测试样本
'''
def oof_funtion(clf,X,y,X_test):
    oof_train=np.zeros((train_len,))
    oof_test=np.zeros((test_len,))
    oof_test_skf=np.empty((NFOLD,test_len))

    for i,(train_index,test_index) in enumerate(kf.split(X)):
        clf.fit(X[train_index],y[train_index])
        oof_train[test_index]=clf.predict(X[test_index])
        oof_test_skf[i,:]=clf.predict(X_test)
    oof_test[:]=np.median(oof_test_skf,axis=0)
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)

et_oof_train, et_oof_test = oof_funtion(et, X, y, X_test) # Extra Trees
rf_oof_train, rf_oof_test = oof_funtion(rf,X, y, X_test) # Random Forest
ada_oof_train, ada_oof_test = oof_funtion(ada, X, y, X_test) # AdaBoost
gb_oof_train, gb_oof_test = oof_funtion(gb,X, y, X_test) # Gradient Boost
svc_oof_train, svc_oof_test = oof_funtion(svc,X, y, X_test) # Support Vector Classifier

X_train=np.concatenate((et_oof_train,rf_oof_train,ada_oof_train,gb_oof_train,svc_oof_train),axis=1)
X_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

oof_train_pd = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
      'SVM' : svc_oof_train.ravel(),
      'GradientBoost': gb_oof_train.ravel(),
    'Survived':train['Survived']
    })

oof_test_pd=pd.DataFrame( {'RandomForest': rf_oof_test.ravel(),
     'ExtraTrees': et_oof_test.ravel(),
     'AdaBoost': ada_oof_test.ravel(),
      'SVM' : svc_oof_test.ravel(),
      'GradientBoost': gb_oof_test.ravel(),
    })

oof_train_pd.to_csv('oof_train_pd.csv')
oof_test_pd.to_csv('oof_test_pd.csv')


import xgboost as xgb

clf_stack = xgb.XGBClassifier(
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 gamma=0.9,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 scale_pos_weight=1)
clf_stack = clf_stack.fit(X_train, y)

'''
结果调优
1，特征选择与处理
2，out of prediction 选择分类器（个数，参数，那些分类器），oof规则
3，最终分类器选择，参数
'''
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf_stack, X_train, y, cv=5)
print(scores)

stack_pred = clf_stack.predict(X_test)

submit = pd.DataFrame({'PassengerId' : test.loc[:,'PassengerId'],
                       'Survived': stack_pred.T})
submit.to_csv("submit.csv", index=False)