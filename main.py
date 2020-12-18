# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


from mlens.ensemble import SuperLearner



# Function responsible for checking our model's performance on the test data
def testSetResultsClassifier(classifier, x_test, y_test, model_name =''):
    predictions = classifier.predict(x_test)
    
    results = []
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    
    results.append(f1)
    results.append(precision)
    results.append(recall)
    results.append(roc_auc)
    results.append(accuracy)
    
    print("\n\n#---------------- Test set results ({}) ----------------#\n".format(model_name))
    print("F1 score, Precision, Recall, ROC_AUC score, Accuracy:")
    print(results)
    
    return results

def saveResult(model, X_test, output='results.csv'):
    """
    CHOOSE MODEL AND SAVE RESULTS
    """
    res = (model.predict(X_test)).astype(int)
    pdResult = pd.DataFrame(list(zip(passId, res)), columns=['PassengerId', 'Survived'])
    
    pdResult.to_csv(output, index=False)
    print("saved {} rows".format(pdResult.shape[0]))



def encodeColumns(df,categorical_cols):
    """
    LABEL ENCODER
    """
    
    le = LabelEncoder()
    return df[categorical_cols].apply(lambda col: le.fit_transform(col))

def alterColumns(df):
    df = df.drop(['PassengerId', 'Cabin', 'Name', 'Ticket'], axis=1)
    
    
    categorical_cols = ['Sex', 'Embarked']
    
    df[categorical_cols] = encodeColumns(df,categorical_cols)
    
    """
    FILL NA
    """
    
    df['Age'].fillna((df['Age'].mean()), inplace=True)
    df['Fare'].fillna((df['Fare'].mean()), inplace=True)
    
    
    
    """
    CREATE NEW COLUMNS
    """
    df['SibsParch'] = df['SibSp'] > 0
    
    df['ClassBinding'] = df['Pclass'] < 3
    
    df['FemaleClassBinding'] = df['Sex'] * df['ClassBinding']
    
    return df

"""
CLEAN DATA
"""
pathTrain = os.path.join('data', 'train.csv')
pathTest = os.path.join('data', 'test.csv')
df = pd.read_csv(pathTrain)
dfTest = pd.read_csv(pathTest)

passId = dfTest.values[:,0]



df = df.dropna()


df = alterColumns(df)
dfTest = alterColumns(dfTest)

"""
STANDARDIZE COLUMNS
"""
sc = StandardScaler()
sc.fit(df['Fare'].values.reshape(-1,1))
dfTest['Fare'] = sc.transform(dfTest['Fare'].values.reshape(-1,1))
df['Fare'] = sc.transform(df['Fare'].values.reshape(-1,1))


"""
MODEL

"""




X = df.loc[:,df.columns != 'Survived']
y = df.loc[:,'Survived']

"""
X['fareq'] = pd.qcut(X['Fare'], q=5).cat.codes
X['Age'] = pd.qcut(X['Age'], q=7).cat.codes

dfTest['fareq'] = pd.qcut(dfTest['Fare'], q=5).cat.codes
dfTest['Age'] = pd.qcut(dfTest['Age'], q=7).cat.codes
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

    knn = KNeighborsClassifier(n_neighbors = 9,
                               p=2)
    
    
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
    
    
    
    rf = RandomForestClassifier(max_depth=None,
                                n_estimators = 30,
                                criterion='entropy')
    
    lr = LogisticRegression(penalty='l2',
                        dual=False,
                        C=3,
                        class_weight = 'uniform',
                        random_state = 1,
                        solver = 'lbfgs')
    

    
    
        
    """
    TRAIN, PREDICT AND RESULTS
    """
    tree = tree.fit(X_train, y_train)
    bag = bag.fit(X_train, y_train)
    ada = ada.fit(X_train, y_train)
    rf = rf.fit(X_train, y_train)
    knn = knn.fit(X_train, y_train)
    lr = lr.fit(X_train, y_train)
           
           
    sl = SuperLearner()
    sl.add(tree).add(bag).add(ada).add(rf)
    
    voting = VotingClassifier(estimators=[('tree', tree),
                                          ('bag', bag),
                                          ('ada', ada),
                                          ('rf', rf),
                                          ('knn', knn)],
                              voting='soft')
    
    gradient = GradientBoostingClassifier(random_state=1)
    
    testSetResultsClassifier(tree, X_test, y_test, 'tree')
    testSetResultsClassifier(bag, X_test, y_test, 'baggings')
    testSetResultsClassifier(ada, X_test, y_test, 'adaboost')
    testSetResultsClassifier(rf, X_test, y_test, 'random forests')
    testSetResultsClassifier(knn, X_test, y_test, 'knn')
    testSetResultsClassifier(lr, X_test, y_test, 'logistic regression')
    testSetResultsClassifier(sl.fit(X_train, y_train), X_test, y_test, 'super learner')
    testSetResultsClassifier(voting.fit(X_train, y_train), X_test, y_test, 'voting classifier')
    testSetResultsClassifier(gradient.fit(X_train, y_train), X_test, y_test, 'gradient classifier')
    #0.8378378378378378
    #saveResult(sl, dfTest, 'sl.csv')
    
    



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
    




    









