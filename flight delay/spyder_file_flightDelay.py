# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 12:44:35 2018

@author: shrey
"""


import numpy as np
import pandas as pd
headers = ['Canceled','Month','DepartureTime','UniqueCarrier','SchedElapsedTime','ArrDelay','DepDelay','Distance']
dataset=pd.read_csv("Data_PredictingFlightDelays.csv",names=headers)
#dataset.describe()
dataset_2=dataset.copy()
from scipy import stats
dataset_2['SchedElapsedTime_transform'] = stats.boxcox(dataset_2['SchedElapsedTime'])[0]

dataset_2.head()
dataset_2.drop(['ArrDelay','DepDelay','Distance','SchedElapsedTime'],axis=1,inplace=True)
#dataset_2.head()
dataset_2.describe()

dataset_2 = pd.get_dummies(dataset_2)

def clean_data(dataset):
    # Create a copy of the original dataframe so the original is not modified
    dataset_2 = dataset.copy()
    
    # Transform SchedElapsedTime
    dataset_2['SchedElapsedTime_transform'] = stats.boxcox(dataset_2['SchedElapsedTime'])[0]

    # Drop unnecessary columns
    dataset_2.drop(['ArrDelay','DepDelay','Distance','SchedElapsedTime'],axis=1,inplace=True)
    
    # Create dummy variables
    dataset_2 = pd.get_dummies(dataset_2)
    
    # Return cleaned dataframe
    return dataset_2

dataset_1=clean_data(dataset)
test = (dataset_1 == dataset_2)
test.drop_duplicates(inplace=True)
test.head()

from sklearn.model_selection import train_test_split
y = dataset_1['Canceled']
X = dataset_1.drop('Canceled',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=888)
"""
StratifiedKFold and LogisticRegressionCV
"""
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV

lr_model = LogisticRegressionCV(cv= None,class_weight='balanced',random_state=888)
lr_model.fit(X_train,y_train)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, confusion_matrix, classification_report

def print_eval_scores(model, X_train, y_train, cv):
    accuracy = cross_val_score(model, X_train, y_train, cv=cv,scoring='accuracy')
    log_loss = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_log_loss')
    roc_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')

    print('Mean Accuracy: %s' % accuracy.mean())
    print('Mean Log Loss: %s' % log_loss.mean())
    print('Mean Area Under ROC Curve: %s' % roc_auc.mean())
    
print_eval_scores(lr_model, X_train, y_train,None)
lr_model.score(X_test,y_test)
y_pred_lr = lr_model.predict(X_test)
import seaborn as sns

print(classification_report(y_test, y_pred_lr))
lr_cm = confusion_matrix(y_test,y_pred_lr)

ax = sns.heatmap(lr_cm,annot=True,cmap='coolwarm',fmt='.0f')


import pickle
lr_model_pickled = pickle.dumps(lr_model)


"""LOGISTIC REGRESSION



"""

from sklearn.ensemble import RandomForestClassifier
rfc_model = RandomForestClassifier(n_estimators=100, random_state=888,class_weight='balanced')
rfc_model.fit(X_train,y_train)

print_eval_scores(rfc_model, X_train, y_train, None)
rfc_model.score(X_test,y_test)

y_pred_rfc = rfc_model.predict(X_test)
print(classification_report(y_test, y_pred_rfc))

rfc_cm = confusion_matrix(y_test,y_pred_rfc)

ax = sns.heatmap(rfc_cm,annot=True,cmap='coolwarm',fmt='.0f')
ax.set_title('Random Forest Classifier Confusion Matrix')
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')


from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train,y_train)

print_eval_scores(knn_model, X_train, y_train, None)

knn_model.score(X_test,y_test)
y_pred_knn = knn_model.predict(X_test)

print(classification_report(y_test, y_pred_rfc))

knn_cm = confusion_matrix(y_test,y_pred_knn)

ax = sns.heatmap(knn_cm,annot=True,cmap='coolwarm',fmt='.0f')
ax.set_title('K Nearest Neighbors Confusion Matrix')
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')


"""
MODEL SELECTION AND CAPTION
"""
def predict_cancellations(df):
    # Clean data
    df1 = clean_data(df)
    
    # Drop 'Canceled' column if it exists
    if 'Canceled' in df1.columns:
        df1.drop('Canceled',axis=1,inplace=True)
        
    # Load pickled logistic regression model
    clf = pickle.loads(lr_model_pickled)
    
    # Generate and return predictions
    y_pred = clf.predict(df1)
    return y_pred

y_pred_func = predict_cancellations(dataset)

y_pred_func
dataset['Predictions'] = y_pred_func

dataset.head()
dataset.to_csv('flight_delay_predictions.csv',index=False)