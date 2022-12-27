#%%
import numpy as np
import pandas as pd

import matplotlib as mpl
#plt.style.use(['dark_background'])
import matplotlib.pyplot as plt
import seaborn as sns
from dataprep.eda import create_report

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import learning_curve
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold

from imblearn.over_sampling import SMOTE

import pymysql
from sqlalchemy import create_engine

import warnings
warnings.filterwarnings('ignore')

from md_sql import pullFromDB
from md import playML, roc_curve_plot, learning_curve_plot

# %%
password = '********'
dbName = 'heartDB'
tableName = 'heart_disease'
df = pullFromDB(password, dbName, tableName)
df.head()
#%%
print(f"Heart Disease df shape: {df.shape}")
print(f"### df info ###")
df.info()
# %%
#dataprep
# powerReport = create_report(df)
# powerReport.save('HeartDisease_summary.html')
# %%
df.columns
# %%
#Process
### 1. 결측치 처리(완료)
### 2. 레이블링 및 더미처리
### 3. 파생변수 생수 
### 4. 스케일링
### 5. 데이터 분리
### 6. 학습 및 평가
#%%
#HeartDiseaseorAttack > target
#컬럼명 소문자로 변경
df.rename(columns={'HeartDiseaseorAttack':'target'}, inplace=True)
df.columns = df.columns.str.lower()
# %%
#dataframe columns
df.columns
# %%
#target 분포 확인
total = len(df['target'])
hd0 = len(df[df['target']==0])
hd1 = len(df[df['target']==1])
print(f"Heart disease 0: {(hd0/total)*100:.2f}%")
print(f"Heart disease 1: {(hd1/total)*100:.2f}%")
# %%
#target countplot(0, 1)
sns.countplot(data=df, x='target')
# %%
cols = list(df.columns)
print(cols)

#%%
objCols = ['target', 'highbp', 'highchol', 'cholcheck', 'smoker', 'stroke', 'diabetes', 'physactivity', 'fruits', 'veggies', 'hvyalcoholconsump', 'anyhealthcare', 'nodocbccost', 'genhlth', 'menthlth', 'physhlth', 'diffwalk', 'sex', 'age', 'education', 'income']

for col in objCols: 
    df = df.astype({f"{col}":'int'}).astype({f"{col}":'category'})

# %%
df.info()
# %%
df.head()
# %%
objCols = ['highbp', 'highchol', 'cholcheck', 'smoker', 'stroke', 'diabetes', 'physactivity', 'fruits', 'veggies', 'hvyalcoholconsump', 'anyhealthcare', 'nodocbccost', 'genhlth', 'menthlth', 'physhlth', 'diffwalk', 'sex', 'age', 'education', 'income']
len(objCols)
#%%
plt.figure(figsize=(20,20))
for idx, col in enumerate(objCols):
    plt.subplot(5, 4, idx+1)
    sns.histplot(data=df, x=col, y='target')
    plt.title(f"{col}")
plt.tight_layout()
plt.show()

# %%
objCols = ['target', 'highbp', 'highchol', 'cholcheck', 'smoker', 'stroke', 'diabetes', 'physactivity', 'fruits', 'veggies', 'hvyalcoholconsump', 'anyhealthcare', 'nodocbccost', 'genhlth', 'menthlth', 'physhlth', 'diffwalk', 'sex', 'age', 'education', 'income']

for col in objCols: 
    df = df.astype({f"{col}":'category'}).astype({f"{col}":'float'})
# %%
df.info()
# %%
plt.figure(figsize=(10,10))
mask = np.zeros_like(df.corr(), dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(df.corr(), cmap='coolwarm_r', mask=mask, vmin = -1, vmax = 1)
# %%
dfCorr = pd.DataFrame(df.corr().iloc[:,0])
dfCorr

#%%
###스케일링
#원본 Copy
dfo = df.copy()
# %%
mms = MinMaxScaler()
df['bmi'] = mms.fit_transform(df[['bmi']])
# %%
df.head()
# %%
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_train.shape}")
# %%
from sklearn.metrics import classification_report
from sklearn.metrics import *

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost
# %%
#모델정의
lr = LogisticRegression(random_state=0)
rfc = RandomForestClassifier(random_state=0)
xgb = xgboost.XGBClassifier(random_state=0)


# %%
models = [lr, rfc, xgb]

plt.figure(figsize=(20,30))
for idx, model in enumerate(models):
    playML(model, X_train, X_test, y_train, y_test)
    plt.subplot(3, 2, (idx+1)*2-1)
    roc_curve_plot(model, X_train, X_test, y_train, y_test)
    plt.subplot(3, 2, (idx+1)*2)
    learning_curve_plot(model, X_train, y_train, 5, 3)
plt.show()
# %%
## StratifiedKFold & TSNE
skf = StratifiedKFold(n_splits=40, random_state=None, shuffle=False)
for idx, (tridx, teidx) in enumerate(skf.split(X, y)):
    if idx == 0:
        X_train, X_test = X.iloc[tridx], X.iloc[teidx]
        y_train, y_test = y.iloc[tridx], y.iloc[teidx]
    else:
        pass

#%%
tsne = TSNE(n_components=2)
tdf = tsne.fit_transform(X_test)
tdf

#%%
plt.scatter(tdf[:,0], tdf[:,1], c=(y_test==0))
plt.scatter(tdf[:,0], tdf[:,1], c=(y_test==1))
# %%
Xi = X.drop(['bmi'], axis=1)
Xi = Xi.astype('int')

X = pd.concat([Xi, X['bmi']], axis=1)
#%%
### SMOTE(오버샘플링)
sm = SMOTE()
X_res, y_res = sm.fit_resample(X, y)
y_res.value_counts()

# %%
#SMOTE를 활용한 dataframe으로 data split
X_res_train, X_res_test, y_res_train, y_res_test = train_test_split(X_res, y_res, test_size=0.2, random_state=0)
print(f"X_res_train shape: {X_res_train.shape}")
print(f"X_res_test shape: {X_res_test.shape}")
print(f"y_res_train shape: {y_res_train.shape}")
print(f"y_res_test shape: {y_res_test.shape}")
# %%
#SMOTE
#머신러닝 수행
models = [lr, rfc, xgb]

plt.figure(figsize=(20,30))
for idx, model in enumerate(models):
    playML(model, X_res_train, X_res_test, y_res_train, y_res_test)
    plt.subplot(3, 2, (idx+1)*2-1)
    roc_curve_plot(model, X_res_train, X_res_test, y_res_train, y_res_test)
    plt.subplot(3, 2, (idx+1)*2)
    learning_curve_plot(model, X_res_train, y_res_train, 5, 3)
plt.show()
# %%
skf = StratifiedKFold(n_splits=40, random_state=None, shuffle=False)
for idx, (tridx, teidx) in enumerate(skf.split(X, y)):
    if idx == 0:
        X_res_train, X_res_test = X.iloc[tridx], X.iloc[teidx]
        y_res_train, y_res_test = y.iloc[tridx], y.iloc[teidx]
    else:
        pass

#%%
tsne = TSNE(n_components=2)
tdf = tsne.fit_transform(X_res_test)
tdf

#%%
plt.scatter(tdf[:,0], tdf[:,1], c=(y_res_test==0))
plt.scatter(tdf[:,0], tdf[:,1], c=(y_res_test==1))
# %%
dfo['target'].value_counts()
# %%
###Under Sampling
dfo = dfo.sample(frac=1)
dfo
# %%
apa = dfo[dfo['target']==1]
anapa = dfo[dfo['target']==0][:23893]
uds_df = pd.concat([apa, anapa], axis=0).sample(frac=1)
#%%
print(f"uds_df shape: {uds_df.shape}")
#%%
X_uds = uds_df.iloc[:,1:]
y_uds = uds_df['target']

X_uds_train, X_uds_test, y_uds_train, y_uds_test = train_test_split(X_uds, y_uds, test_size=0.2, random_state=0)
print(f"X_uds_train shape: {X_uds_train.shape}")
print(f"X_uds_test shape: {X_uds_test.shape}")
print(f"y_uds_train shape: {y_uds_train.shape}")
print(f"y_uds_test shape: {y_uds_test.shape}")

# %%
models = [lr, rfc, xgb]

plt.figure(figsize=(20,30))
for idx, model in enumerate(models):
    playML(model, X_uds_train, X_uds_test, y_uds_train, y_uds_test)
    plt.subplot(3, 2, (idx+1)*2-1)
    roc_curve_plot(model, X_uds_train, X_uds_test, y_uds_train, y_uds_test)
    plt.subplot(3, 2, (idx+1)*2)
    learning_curve_plot(model, X_uds_train, y_uds_train, 5, 3)
plt.show()
#%%
tsne = TSNE(n_components=2)
tdf = tsne.fit_transform(X_uds)
tdf


#%%
plt.scatter(tdf[:,0], tdf[:,1], c=(y_uds==0))
plt.scatter(tdf[:,0], tdf[:,1], c=(y_uds==1))
# %%
