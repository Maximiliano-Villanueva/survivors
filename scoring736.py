# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from mlens.ensemble import SuperLearner

import matplotlib.pyplot as plt
import seaborn as sn

import sys

"""
CLEAN DATA
"""
pathTrain = os.path.join('data', 'train.csv')
pathTest = os.path.join('data', 'test.csv')
df = pd.read_csv(pathTrain)
dfTest = pd.read_csv(pathTest)

passId = dfTest.values[:,0]



df = df.dropna()

df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
dfTest = dfTest.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)


"""
LABEL ENCODER
"""

le = LabelEncoder()

categorical_cols = ['Sex', 'Embarked']

#female = 0, male = 1
df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))

le = LabelEncoder()
dfTest[categorical_cols] = dfTest[categorical_cols].apply(lambda col: le.fit_transform(col))



"""
FILL NA
"""

dfTest['Age'].fillna((dfTest['Age'].mean()), inplace=True)
dfTest['Fare'].fillna((dfTest['Fare'].mean()), inplace=True)


"""
STANDARDIZE COLUMNS
"""
sc = StandardScaler()
sc.fit(df['Fare'].values.reshape(-1,1))
dfTest['Fare'] = sc.transform(dfTest['Fare'].values.reshape(-1,1))
df['Fare'] = sc.transform(df['Fare'].values.reshape(-1,1))



"""
CREATE NEW COLUMNS
"""
df['SibsParch'] = df['SibSp'] > 0

dfTest['SibsParch'] = dfTest['SibSp'] > 0


df['ClassBinding'] = df['Pclass'] < 3

dfTest['ClassBinding'] = dfTest['Pclass'] < 3

df['FemaleClassBinding'] = df['Sex'] * df['ClassBinding']

dfTest['FemaleClassBinding'] = ((dfTest['Sex']-1)*-1) * dfTest['ClassBinding']




#print(df.columns)
#print(df.head())


"""
MODEL

"""

"""
sn.heatmap(df.corr(), annot=True)
plt.show()
"""



X = df.loc[:,df.columns != 'Survived']
y = df.loc[:,'Survived']

"""
X['fareq'] = pd.qcut(X['Fare'], q=10).cat.codes
X['Age'] = pd.qcut(X['Age'], q=7).cat.codes
"""

X['fareq'] = pd.qcut(X['Fare'], q=10).cat.codes
X['Age'] = pd.qcut(X['Age'], q=7).cat.codes

dfTest['fareq'] = pd.qcut(dfTest['Fare'], q=10).cat.codes
dfTest['Age'] = pd.qcut(dfTest['Age'], q=7).cat.codes


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y)



"""
EMBEDDING MODELS
"""
if True:
    tree = DecisionTreeClassifier(criterion='entropy', 
                                  max_depth=None,
                                  random_state=1)
    
    bag = BaggingClassifier(base_estimator=tree,
                            n_estimators=100, 
                            max_samples=1.0, 
                            max_features=1.0, 
                            bootstrap=True, 
                            bootstrap_features=False, 
                            n_jobs=1, 
                            random_state=1)
    
    
    ada = AdaBoostClassifier(base_estimator=tree,
                             n_estimators=100, 
                             learning_rate=0.1,
                             random_state=1)
    
    rf_params = {"rf__n_estimators" : [10,30, 50 , 80 , 100],
                 "rf__criterion" : ["gini", "entropy"]}
    
    rf = RandomForestClassifier(max_depth=None)
    
    
    search = GridSearchCV(Pipeline(steps=[('rf',rf)]), rf_params)
    search.fit(X_train, y_train)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)
    
    
    
    
    
    
    
    """
    PREDICT, TRAIN AND RESULTS
    """
    
    """
    tree = tree.fit(X_train, y_train)
    pred = tree.predict(X_test)
    
    acc = accuracy_score(y_test, pred)
    print('lulitree: ', acc)
    
    bag = bag.fit(X_train, y_train)
    pred = bag.predict(X_test)
    
    acc = accuracy_score(y_test, pred)
    print('lulibag: ', acc)
    
    
    
    ada = ada.fit(X_train, y_train)
    pred = ada.predict(X_test)
    
    acc = accuracy_score(y_test, pred)
    print('luliada: ', acc)
    
    pred = search.predict(X_test)
    
    acc = accuracy_score(y_test, pred)
    print('RandomForest: ', acc)
    """
    
    
    sl = SuperLearner()
    sl.add(tree).add(ada).add(search).add(search)
    
    sl.fit(X_train, y_train)
    
    pred = sl.predict(X_test)
    
    acc = accuracy_score(y_test, pred)
    print('sl: ', acc)
    
    if True:
        bag_result = (sl.predict(dfTest)).astype(int)
        pdResult = pd.DataFrame(list(zip(passId, bag_result)), columns=['PassengerId', 'Survived'])
        
        pdResult.to_csv('results.csv', index=False)


"""
PCA EXAMPLE
"""
if False:

    tree = DecisionTreeClassifier(criterion='entropy', 
                                  max_depth=None,
                                  random_state=1)
    
    
    ada = AdaBoostClassifier(base_estimator=tree,
                             n_estimators=100, 
                             learning_rate=0.1,
                             random_state=1)
    
    pipe = Pipeline(steps=[('pca',PCA()), ('ada',ada)])
    
    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    param_grid = {
        'pca__n_components': [3, 4, 5, 6, 7, 8, 9],
    }
    search = GridSearchCV(pipe, param_grid)
    search.fit(X_train, y_train)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)
    
    pred = search.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print('luliada: ', acc)
    


"""
SVM MODEL
"""
if False:
    
    pipe = Pipeline(steps=[('pca',PCA()), ('svc',SVC(class_weight='balanced'))])
    
    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    param_grid = {
        'pca__n_components': [3, 4, 5, 6, 7, 8, 9],
        'svc__C' : [1000],
        'svc__gamma' : [0.01],
        'svc__kernel' : [ 'rbf']
    }
    
    clf = GridSearchCV(pipe, param_grid)
    clf.fit(X_train, y_train)
    

    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print('luliada: ', acc)

"""
CHOOSE MODEL AND SAVE RESULTS
"""
if False:
    bag_result = ada.predict(dfTest)
    pdResult = pd.DataFrame(list(zip(passId, bag_result)), columns=['PassengerId', 'Survived'])
    
    pdResult.to_csv('results.csv', index=False)


    









