import numpy as np
import scipy as sp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score 
import pickle
import joblib
import itertools


# This part is irrelevant to the cross-validation
pred_word=[]
def _get_top_probabilities(classifier, pred_prob_arr):
    return dict(sorted(dict(zip(classifier.classes_,pred_prob_arr)).items(), key=lambda x:x[1], reverse=True))
    
def get_top_digraphs(classifier,pred_prob_arr, no_of_digraphs=10):
    return list(_get_top_probabilities(classifier,pred_prob_arr).keys())[:no_of_digraphs]

def generate_words(arr_of_digraphs):
    for digraph_tuple in itertools.product(*arr_of_digraphs):
         pred_word.append(''.join(digraph_tuple))

msu_dataset=pd.read_csv("msuupdated.csv")
stony_dataset = pd.read_csv("stonybrooksdataset_updated.csv")
greycweb_dataset = pd.read_csv("greycwebdata.csv")
greyc_normal_dataset = pd.read_csv("greyc_normal.csv")
dataframe=pd.concat([greycweb_dataset, greyc_normal_dataset])

X=dataframe.iloc[:,2:].values
y=dataframe.iloc[:,1].values

scaler = StandardScaler()
X_new = scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X_new,y,test_size=0.2,random_state=61)

classifier = RandomForestClassifier(random_state = 23)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print ("Accuracy Score : {}%".format(accuracy_score(y_test, y_pred)*100))


#Cross-Validation
scores = [] #Keeps track of the accuracy scores
clf = RandomForestClassifier(random_state = 23)
cv = KFold(n_splits=3, random_state=42, shuffle=True) # The number of folds can be changed here
for train_index, test_index in cv.split(X_new):
    X_train, X_test, y_train, y_test = X_new[train_index], X_new[test_index], y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred)*100)
    
print(np.mean(scores))
print(scores)










