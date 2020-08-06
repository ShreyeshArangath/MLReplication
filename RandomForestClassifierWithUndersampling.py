import numpy as np
import scipy as sp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Importing datasets
msu_dataset=pd.read_csv("msuupdated.csv")
stony_dataset = pd.read_csv("stonybrooksdataset_updated.csv")
greycweb_dataset = pd.read_csv("greycwebdata.csv")
greyc_normal_dataset = pd.read_csv("greyc_normal.csv")

# Splitting the datasets for undersampling 
dataframe=pd.concat([ msu_dataset, greycweb_dataset, greyc_normal_dataset])
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

# Classification
final_dataframe = pd.concat([undersampled_dataframe, dataframe_with_less_than_1000_samples])

X=final_dataframe.iloc[:,1:].values
y=final_dataframe.iloc[:,0].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=61)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = RandomForestClassifier(random_state = 23)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print ("Accuracy Score : {}%".format(accuracy_score(y_test, y_pred)*100))

