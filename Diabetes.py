#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np
import seaborn as sns
#Seaborn comes with a number of customized themes and a high-level interface for controlling the look of matplotlib figures.
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[56]:


df=pd.read_csv('diabetes.csv')
column = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
df.columns=column
df.head()


# In[57]:


df.describe()


# In[58]:


df.info()


# In[59]:


print((df[['Glucose']] == 0).sum())
print((df[['BloodPressure']] == 0).sum())
print((df[['SkinThickness']] == 0).sum())
print((df[['Insulin']] == 0).sum())
print((df[['BMI']] == 0).sum())


# In[60]:


df.isnull().values.any()


# In[61]:


df.hist(bins=10,figsize=(10,10))
plt.show()


# In[62]:


# mark zero values as missing or NaN
df[['Glucose']] = df[['Glucose']].replace(0, np.NaN)
df[['BloodPressure']] = df[['BloodPressure']].replace(0, np.NaN)
df[['SkinThickness']] = df[['SkinThickness']].replace(0, np.NaN)
df[['Insulin']] = df[['Insulin']].replace(0, np.NaN)
df[['BMI']] = df[['BMI']].replace(0, np.NaN)
# count the number of NaN values in each column
print(df.isnull().sum())


# In[63]:


df.head()


# In[64]:


df.fillna(df.mean(), inplace=True)
print(df.isnull().sum())


# In[65]:


df.head()


# In[66]:


df.hist(bins=10,figsize=(10,10))
plt.show()


# In[67]:


#glucose levels, age, BMI and number of pregnancies all have significant correlation
#with the outcome variable. Also notice the correlation between pairs of features, like age and pregnancies, or insulin and 
#skin thickness.
sns.heatmap(df.corr())


# In[68]:


sns.countplot(y=df['Outcome'],palette='Set1')


# In[69]:


sns.set(style="ticks")#
sns.pairplot(df, hue="Outcome")


# In[70]:


sns.set(style="whitegrid")
df.boxplot(figsize=(15,6))


# In[71]:


sns.set(style="whitegrid")


sns.set(rc={'figure.figsize':(4,2)})
sns.boxplot(x=df['Insulin'])
plt.show()
sns.boxplot(x=df['BloodPressure'])
plt.show()
sns.boxplot(x=df['DiabetesPedigreeFunction'])
plt.show()


# In[72]:


Q1=df.quantile(0.25)
Q3=df.quantile(0.75)
IQR=Q3-Q1

print("---Q1--- \n",Q1)
print("\n---Q3--- \n",Q3)
print("\n---IQR---\n",IQR)


# In[73]:


df_out = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
df.shape,df_out.shape


# In[74]:


sns.set(style="ticks")
sns.pairplot(df_out, hue="Outcome")
plt.show()


# In[75]:


X=df_out.drop(columns=['Outcome'])
y=df_out['Outcome']


# In[76]:


from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2)


# In[77]:


train_X.shape,test_X.shape,train_y.shape,test_y.shape


# In[78]:


from sklearn.metrics import confusion_matrix,accuracy_score,make_scorer
from sklearn.model_selection import cross_validate

def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]

#cross validation purpose
scoring = {'accuracy': make_scorer(accuracy_score),'prec': 'precision'}
scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
           'fp': make_scorer(fp), 'fn': make_scorer(fn)}

def display_result(result):
    print("TP: ",result['test_tp'])
    print("TN: ",result['test_tn'])
    print("FN: ",result['test_fn'])
    print("FP: ",result['test_fp'])


# In[79]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import confusion_matrix, classification_report

acc=[]
roc=[]

clf=LogisticRegression()
clf.fit(train_X,train_y)
y_pred=clf.predict(test_X)
#find accuracy
ac=accuracy_score(test_y,y_pred)
acc.append(ac)

#find the ROC_AOC curve
rc=roc_auc_score(test_y,y_pred)
roc.append(rc)
print("\nAccuracy {0} ROC {1}".format(ac,rc))

#cross val score
result=cross_validate(clf,train_X,train_y,scoring=scoring,cv=10)
display_result(result)

cf=confusion_matrix(test_y, y_pred)
clf_rp=classification_report(test_y, y_pred)
print(cf)
print(clf_rp)

#display predicted values uncomment below line
#pd.DataFrame(data={'Actual':test_y,'Predicted':y_pred}).head()


# In[80]:


#Support Vector Machine
#ROC-Reciever operating charestric   
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

clf=SVC(kernel='linear')
clf.fit(train_X,train_y)
y_pred=clf.predict(test_X)
#find accuracy
ac=accuracy_score(test_y,y_pred)
acc.append(ac)

#find the ROC_AOC curve
rc=roc_auc_score(test_y,y_pred)
roc.append(rc)
print("\nAccuracy {0} ROC {1}".format(ac,rc))

#cross val score
result=cross_validate(clf,train_X,train_y,scoring=scoring,cv=10)
display_result(result)

cf=confusion_matrix(test_y, y_pred)
clf_rp=classification_report(test_y, y_pred)
print(cf)
print(clf_rp)

#display predicted values uncomment below line
#pd.DataFrame(data={'Actual':test_y,'Predicted':y_pred}).head()


# In[81]:


#Naive Bayes Theorem
#import library
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report

clf=GaussianNB()
clf.fit(train_X,train_y)
y_pred=clf.predict(test_X)
#find accuracy
ac=accuracy_score(test_y,y_pred)
acc.append(ac)

#find the ROC_AOC curve
rc=roc_auc_score(test_y,y_pred)
roc.append(rc)
print("\nAccuracy {0} ROC {1}".format(ac,rc))

#cross val score
result=cross_validate(clf,train_X,train_y,scoring=scoring,cv=10)
display_result(result)
cf=confusion_matrix(test_y, y_pred)
clf_rp=classification_report(test_y, y_pred)
print(cf)
print(clf_rp)
#display predicted values uncomment below line
#pd.DataFrame(data={'Actual':test_y,'Predicted':y_pred}).head()


# In[82]:


ax=plt.figure(figsize=(9,3))
plt.bar(['Logistic Regression','SVM','Naivye Bayes'],acc,label='Accuracy')
plt.ylabel('Accuracy Score')
plt.xlabel('Algortihms')
plt.show()

ax=plt.figure(figsize=(9,3))
plt.bar(['Logistic Regression','SVM','Naivye Bayes'],roc,label='ROC AUC')
plt.ylabel('ROC AUC')
plt.xlabel('Algortihms')
plt.show()


# In[ ]:




