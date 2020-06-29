# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#importing training set
dataset= pd.read_csv('train.csv')
dataset.info()

#visualization
sns.set_style('whitegrid')
sns.countplot(x='Survived', data=dataset, palette= 'RdBu_r')
sns.countplot(x='Survived', hue= 'Sex' , data= dataset, palette= 'RdBu_r' )
sns.countplot(x='Survived',hue= 'Pclass', data= dataset, palette ='rainbow')
sns.distplot( dataset['Age'].dropna(),color = 'darkred',bins=30)

dataset['Fare'].hist(color = 'green',bins= 40, figsize=(8,4))

plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass', y='Age', data=dataset, palette= 'winter')



#function to assign age
def impute_age(cols):
  Age = cols[0]
  Pclass = cols[1]
  if pd.isnull(Age):
     if Pclass == 1:
        return 37
     elif Pclass == 2:
        return 29
     else:
        return 24
  else:
      return Age
dataset['Age'] = dataset[['Age', 'Pclass']].apply(impute_age, axis = 1)



def impute_cabin(col):
  Cabin = col[0]
  if type(Cabin) == str:
    return 1
  else:
    return 0
 
dataset['Cabin'] = dataset[['Cabin']].apply(impute_cabin, axis = 1)



dataset1 = dataset
 
sex = pd.get_dummies(dataset1['Sex'],drop_first=True)
embark = pd.get_dummies(dataset1['Embarked'],drop_first=True)
dataset1.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
dataset1 = pd.concat([dataset1,sex,embark],axis=1)

X= dataset1.iloc[:,:-1].values
y= dataset1.iloc[:,1].values



#working on test set
dataset2= pd.read_csv('test.csv')


dataset2.info()


def impute_age(cols):
  Age = cols[0]
  Pclass = cols[1]
  if pd.isnull(Age):
     if Pclass == 1:
        return 37
     elif Pclass == 2:
        return 29
     else:
        return 24
  else:
      return Age
dataset2['Age'] = dataset2[['Age', 'Pclass']].apply(impute_age, axis = 1)


def impute_cabin(col):
  Cabin = col[0]
  if type(Cabin) == str:
    return 1
  else:
    return 0

dataset2['Cabin'] = dataset2[['Cabin']].apply(impute_cabin, axis = 1)


mean_value=dataset2['Fare'].mean()

dataset2['Fare']=dataset2['Fare'].fillna(mean_value)

sex = pd.get_dummies(dataset2['Sex'],drop_first=True)
embark = pd.get_dummies(dataset2['Embarked'],drop_first=True)
dataset2.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
dataset2 = pd.concat([dataset2,sex,embark],axis=1)





# Applying logistic regression
regressor = LogisticRegression()
regressor.fit(X,y)
pred = regressor.predict(dataset2)
pred_1=pd.DataFrame(pred)

#converting result to csv file
pred_2= pd.concat([dataset2['PassengerId'],pred_1], axis=1)
pred_2.to_csv('results.csv', index=False)






