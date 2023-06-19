#!/usr/bin/env python
# coding: utf-8

# In[195]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[196]:


df1=pd.read_csv('gender_submission.csv')


# In[197]:


df=pd.read_csv('train.csv')


# In[198]:


df2=pd.read_csv('test.csv')


# In[199]:


df3=pd.merge(df2,df1,on='PassengerId')


# In[200]:


total=pd.concat([df,df3])
total


# In[201]:


total.isnull().sum()


# In[202]:


total=total.drop(['Cabin','Ticket','Name'],axis=1)


# In[203]:


total['Age'].fillna(value=total['Age'].sum(),inplace=True)


# In[204]:


total['Embarked'].fillna(method='ffill',inplace=True)


# In[205]:


total['Fare'].fillna(value=total['Fare'].sum(),inplace=True)


# In[206]:


total=pd.get_dummies(total,drop_first=True)


# In[207]:


total


# In[208]:


from sklearn.model_selection import train_test_split


# In[209]:


y=total['Survived']


# In[210]:


X=total.drop('Survived',axis=1)


# In[211]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# In[212]:


from sklearn.ensemble import RandomForestClassifier


# In[233]:


model = RandomForestClassifier(n_estimators=10,max_features='auto',random_state=101)


# In[234]:


model.fit(X_train,y_train)


# In[235]:


X_train = X_train.astype('float32')


# In[236]:


preds = model.predict(X_test)


# In[ ]:





# In[237]:


from sklearn.tree import DecisionTreeClassifier


# In[238]:


model1 = DecisionTreeClassifier()


# In[239]:


model1.fit(X_train,y_train)


# In[240]:


preds1 = model1.predict(X_test)


# In[241]:


from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix,accuracy_score


# In[242]:


confusion_matrix(y_test,preds)


# In[243]:


confusion_matrix(y_test,preds1)


# In[244]:


print(classification_report(y_test,preds))


# In[245]:


print(classification_report(y_test,preds1))


# In[246]:


test_error = []

for n in range(1,40):
    # Use n random trees
    model = RandomForestClassifier(n_estimators=n,max_features='auto')
    model.fit(X_train,y_train)
    test_preds = model.predict(X_test)
    test_error.append(1-accuracy_score(test_preds,y_test))
 


# In[247]:


plt.figure(figsize=(12,8))
plt.plot(range(1,40),test_error,label='Test Error')
plt.legend()


# In[248]:


model.feature_importances_


# In[249]:


model1.feature_importances_


# In[250]:


sns.pairplot(data=df,hue='Survived')


# In[251]:


plt.figure(figsize=(12,8))
sns.scatterplot(data=df,x='Age',hue='Sex',y='Survived',style='Pclass')


# In[252]:


plot_confusion_matrix(model,X_test,y_test)


# In[ ]:




