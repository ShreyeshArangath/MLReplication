import os
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

dataPath = os.path.join(os.getcwd(),"Data")

def _getDataFilePath(fileName):
    return os.path.join(dataPath+os.sep, fileName)

def _getDigraphFrequencies(dataframe):
    """Returns the digraph frequencies within a database."""
    return dataframe.groupby("digraph").count()

def _perfectTestSplit(dataframe, test_size=0.2):
    """Returns a train dataframe and a perfectly split test dataframe."""
    unique_digraph_set = set(dataframe['digraph']) 
    test_split = 0.2
    min_occurences_perfect_split = int(len(dataframe)*test_split//len(unique_digraph_set))
    train_dataframe = pd.DataFrame()
    test_dataframe = pd.DataFrame()
    
    for digraph in unique_digraph_set: 
        temp_dataframe = dataframe[dataframe['digraph']==digraph]
        temp_dataframe = temp_dataframe.sample(frac=1).reset_index(drop=True)
        temp_test_dataframe = temp_dataframe.iloc[:min_occurences_perfect_split,:]
        temp_train_dataframe = temp_dataframe.iloc[min_occurences_perfect_split:,:]
        train_dataframe = pd.concat([train_dataframe, temp_train_dataframe], ignore_index=True, sort=False)
        test_dataframe = pd.concat([test_dataframe, temp_test_dataframe], ignore_index=True, sort=False)
        
    return train_dataframe, test_dataframe


# Importing datasets
msu_dataset=pd.read_csv(_getDataFilePath("msuupdated.csv"))
stony_dataset = pd.read_csv(_getDataFilePath("stonybrooksdataset_updated.csv"))
greycweb_dataset = pd.read_csv(_getDataFilePath("greycwebdata.csv"))
greyc_normal_dataset = pd.read_csv(_getDataFilePath("greyc_normal.csv"))

# Splitting the datasets for undersampling 
dataframe=pd.concat([ greycweb_dataset, greyc_normal_dataset])
dataframe=dataframe.groupby("digraph").filter(lambda x: len(x) > 100)
dataframe_with_less_than_1000_samples = dataframe.groupby("digraph").filter(lambda x: len(x) < 1000) 
dataframe_to_undersample = dataframe.groupby("digraph").filter(lambda x: len(x) >= 1000)

# Under Sampling 
X=dataframe_to_undersample.iloc[:,2:].values
y=dataframe_to_undersample.iloc[:,1].values
undersample = NearMiss()
X_under, y_under = undersample.fit_resample(X, y)
# To delete the extra index column 
del dataframe_with_less_than_1000_samples['Unnamed: 0'] 
undersampled_dataframe = pd.DataFrame(X_under, columns= ['inter-key','uut', 'ddt'])
undersampled_dataframe.insert(0, "digraph", y_under)

# Preprocessing 
final_dataframe = pd.concat([undersampled_dataframe, dataframe_with_less_than_1000_samples])
_getDigraphFrequencies(final_dataframe)

## Splitting test data as perfectly balanced
train_dataframe, test_dataframe = _perfectTestSplit(final_dataframe)

X_train=train_dataframe.iloc[:,1:].values
y_train=train_dataframe.iloc[:,0].values

X_test=test_dataframe.iloc[:,1:].values
y_test=test_dataframe.iloc[:,0].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Classification
classifier = RandomForestClassifier(random_state = 23)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print ("Accuracy Score : {}%".format(accuracy_score(y_test, y_pred)*100))

