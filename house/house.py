# -*- coding:utf-8 -*-

import pandas as pd
from pandas import read_csv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.stats import skew

train=read_csv('train.csv')
test=read_csv('test.csv')
data=pd.concat([train.drop('SalePrice',axis=1),test],axis=0)


len_train=len(train)
len_test=len(test)
features=data.columns
print '特征为：',features
print '训练数据%d个(%.1f percent)，测试数据%d个(%.1f percent)，特征个数%d个'%(
    len_train,len_train*1./(len_train+len_test)*100,len_test,len_test*1./(len_train+len_test)*100,len(features)
)
# sns.distplot(np.log(train['SalePrice']),kde=False)
# plt.show()

s=data.isnull().sum().sort_values(ascending=False)
s=s[s.values!=0]
f=s.to_frame()
f['dtype']=data.dtypes[s.index].values
f_object=f[f['dtype']=='object']
f_number=f[f['dtype']!='object']
# print 'object 存在NAN:',f_object
# print 'number 存在NAN',f_number
'''
 object 存在NAN:                  0   dtype
PoolQC        2909  object
MiscFeature   2814  object
Alley         2721  object
Fence         2348  object

FireplaceQu   1420  object
GarageCond     159  object
GarageQual     159  object
GarageFinish   159  object
GarageType     157  object

BsmtCond        82  object
BsmtExposure    82  object
BsmtQual        81  object
BsmtFinType2    80  object
BsmtFinType1    79  object
MasVnrType      24  object
MSZoning         4  object
Utilities        2  object
Functional       2  object
Exterior1st      1  object
Exterior2nd      1  object
SaleType         1  object
Electrical       1  object
KitchenQual      1  object

number 存在NAN                 0    dtype
LotFrontage   486  float64
GarageYrBlt   159  float64
MasVnrArea     23  float64
BsmtHalfBath    2  float64
BsmtFullBath    2  float64
BsmtFinSF1      1  float64
BsmtFinSF2      1  float64
BsmtUnfSF       1  float64
TotalBsmtSF     1  float64
GarageArea      1  float64
GarageCars      1  float64

'''

'''
缺失太多去掉此特征，转为bool
PoolQC        2909   object
MiscFeature   2814   object
Alley         2721   object
Fence         2348   object
'''
feats=['PoolQC','MiscFeature','Alley','Fence']
data['MiscFeature_known']=data['MiscFeature'].isnull()==False
data['Alley_known']=data['Alley'].isnull()==False
data['Fence_known']=data['Fence'].isnull()==False
data['FireplaceQu_known']=data['FireplaceQu'].isnull()==False
data.drop(feats,axis=1,inplace=True)

'''
选出数值类型，填充均值
'''
index=f_number.index
# print data[index].mean()
data[index]=data[index].fillna(data[index].mean())

'''
将空值作为一种新的类型
FireplaceQu   1420  object
GarageCond     159  object
GarageQual     159  object
GarageFinish   159  object
GarageType     157  object
'''
l=['FireplaceQu','GarageCond','GarageQual','GarageFinish','GarageType']
data[l]=data[l].fillna('NAN')

'''
填充上一条数据
BsmtCond        82  object
BsmtExposure    82  object
BsmtQual        81  object
BsmtFinType2    80  object
BsmtFinType1    79  object
MasVnrType      24  object
MSZoning         4  object
Utilities        2  object
Functional       2  object
Exterior1st      1  object
Exterior2nd      1  object
SaleType         1  object
Electrical       1  object
KitchenQual      1  object
'''
l=f_object[f[0]<100].index
data[l]=data[l].fillna(method='pad')

# data['MiscFeature_known']=data['MiscFeature_known'].astype('category')
# data['MiscFeature_known'].cat.categories = [0,1]
# data['MiscFeature_known'] = data['MiscFeature_known'].astype("int")


# print data['Fence_known'].dtypes

# print data.info()
# 特征值处理
train['SalePrice']=np.log1p(train['SalePrice'])

numeric_feats = data.dtypes[(data.dtypes != "object")&(data.dtypes!='bool')].index
encode_feats=data.dtypes[(data.dtypes == "object")|(data.dtypes=='bool')].index


# skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
# skewed_feats=skewed_feats[skewed_feats>0.75].index
# print skewed_feats
# data[skewed_feats]=np.log1p(data[skewed_feats])


from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler,LabelBinarizer
# 使用DataFrameMapper dataframe 不能有空值
params=[]

for feature in encode_feats:
    params.append((feature,LabelBinarizer()))
for feature in numeric_feats:
    params.append(([feature],StandardScaler()))

mapper=DataFrameMapper(params)
# print mapper
# print data.head(8)
D=mapper.fit_transform(data)

X=D[:len_train]
X_test=D[len_train:]
y=train['SalePrice'].values

from sklearn.model_selection import cross_val_score,GridSearchCV,cross_val_predict
from sklearn.linear_model import Ridge,Lasso

# alphas = [0.05, 0.1, 0.3, 1, 3, 5,10,8]
#
# for alpha in alphas:
#     print np.mean(cross_val_score(Ridge(alpha=alpha),X,y,cv=5))
'''
0.846384256225
0.849276670027
0.856146609109
0.864408242783
0.87005947539
0.8718224646
0.873215407066
0.872897980196
'''
ridge=Ridge(alpha=8)

# alphas = [0.05, 0.1, 0.3, 0.001,0.005,0.003]
#
# for alpha in alphas:
#     print np.mean(cross_val_score(Lasso(alpha=alpha),X,y,cv=5))

lasso=Lasso(alpha=0.001)
ridge.fit(X,y)
lasso.fit(X,y)
import xgboost as xgb
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(X, y)


ridge_pred=np.expm1(ridge.predict(X_test))
lasso_pred=np.expm1(lasso.predict(X_test))
xgb_pred=np.expm1(model_xgb.predict(X_test))

result=np.array([ridge_pred,lasso_pred,xgb_pred])
result=np.mean(result,axis=0)

submit=pd.DataFrame({
    'Id':test['Id'],
    'SalePrice':result
})
# print submit.head(8)
submit.to_csv("submit.csv", index=False)