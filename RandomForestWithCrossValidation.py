import os
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# This part is irrelevant to the cross-validation
pred_word=[]
def _get_top_probabilities(classifier, pred_prob_arr):
    return dict(sorted(dict(zip(classifier.classes_,pred_prob_arr)).items(), key=lambda x:x[1], reverse=True))
    
def get_top_digraphs(classifier,pred_prob_arr, no_of_digraphs=10):
    return list(_get_top_probabilities(classifier,pred_prob_arr).keys())[:no_of_digraphs]

def generate_words(arr_of_digraphs):
    for digraph_tuple in itertools.product(*arr_of_digraphs):
         pred_word.append(''.join(digraph_tuple))

dataPath = os.path.join(os.getcwd(),"Data")
def _getDataFilePath(fileName):
    return os.path.join(dataPath+os.sep, fileName)

# Importing datasets
msu_dataset=pd.read_csv(_getDataFilePath("msuupdated.csv"))
stony_dataset = pd.read_csv(_getDataFilePath("stonybrooksdataset_updated.csv"))
greycweb_dataset = pd.read_csv(_getDataFilePath("greycwebdata.csv"))
greyc_normal_dataset = pd.read_csv(_getDataFilePath("greyc_normal.csv"))

dataframe=pd.concat([greycweb_dataset, greyc_normal_dataset])

X=dataframe.iloc[:,2:].values
print(X)
y=dataframe.iloc[:,1].values
print(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=61)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = RandomForestClassifier(max_depth=15, random_state = 23)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print ("Accuracy Score : {}%".format(accuracy_score(y_test, y_pred)*100))


#Cross-Validation
scores = [] #Keeps track of the accuracy scores
clf = RandomForestClassifier(max_depth=15, random_state = 23)
cv = KFold(n_splits=10, random_state=42, shuffle=True) # The number of folds can be changed here
for train_index, test_index in cv.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred)*100)
    
print(np.mean(scores))
print(scores)










