import numpy as np
import itertools
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score
from imblearn.under_sampling import NearMiss
from sklearn.preprocessing import StandardScaler
import pickle
import joblib
import RockYouDatasetParser 
import SystemPath

path = SystemPath.Path()
parser = RockYouDatasetParser.RockYouDatasetParser()

def _getDigraphFrequencies(dataframe):
    """Returns the digraph frequencies within a dataframe."""
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

def _getBinCount(np_array):
    classes, indices = np.unique(np_array, return_inverse=True)
    class_counts = np.bincount(indices)
    return class_counts

def _trainTestSplit(final_dataframe, test_size=0.2):
    """Returns X_train, X_test, y_train, y_test. The test data returned is perfectly balanced."""
    
    unique_digraph_set = set(final_dataframe['digraph']) 
    X_train, X_test , y_train , y_test = train_test_split(X, y, test_size=test_size, random_state = 0)
    min_instance_count = np.min(_getBinCount(y_test))
    print("The minimum number of instances required for a perfectly balanced test data: ",min_instance_count)
    
    y_test_balanced = []
    y_test_discard = []
    X_test_balanced = []
    X_test_discard = []
    indices_to_discard = []   
    for digraph in unique_digraph_set: 
        indices = (np.where(y_test==digraph))[0]
        indices_to_keep = indices[:np.min(min_instance_count)]
        indices_to_discard = indices[np.min(min_instance_count):]
        y_test_balanced.extend(np.array(y_test)[indices_to_keep])
        y_test_discard.extend(np.array(y_test)[indices_to_discard])
        X_test_balanced.extend(np.array(X_test)[indices_to_keep])
        X_test_discard.extend(np.array(X_test)[indices_to_discard])
    
    X_test_discard= np.array(X_test_discard)
    X_test_balanced = np.array(X_test_balanced)
    y_test_balanced = np.array(y_test_balanced)
    y_test_discard = np.array(y_test_discard)
    
    X_train_updated = np.concatenate((X_train, X_test_discard))
    y_train_updated = np.concatenate((y_train, y_test_discard))
    
    print("The instances per digraph in testing data: ", _getBinCount(y_test_balanced))
    print("The instances per digraph in training data: ", _getBinCount(y_train_updated))
    
    return X_train_updated, X_test_balanced, y_train_updated, y_test_balanced

    
# Importing datasets
msu_dataset=pd.read_csv(path.getDataFilePath("msuupdated.csv"))
stony_dataset = pd.read_csv(path.getDataFilePath("stonybrooksdataset_updated.csv"))
greycweb_dataset = pd.read_csv(path.getDataFilePath("greycwebdata.csv"))
greyc_normal_dataset = pd.read_csv(path.getDataFilePath("greyc_normal.csv"))
rockYouDataframe = pd.read_csv(path.getDataFilePath("rockyou8subset.csv"))
relevantDigraphDataframe = pd.read_csv(path.getDataFilePath("uniqueDigraphs.csv"))
originalRockYouDataframeWithCount = pd.read_csv(path.getDataFilePath("rockyoudataset.csv"))

# Splitting the datasets for undersampling 
dataframe=pd.concat([ msu_dataset ,greycweb_dataset, greyc_normal_dataset])
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
X=final_dataframe.iloc[:,1:].values
y=final_dataframe.iloc[:,0].values
_getDigraphFrequencies(final_dataframe)

# Scaling features
X_train, X_test, y_train, y_test = _trainTestSplit(final_dataframe, 0.75)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Classification
classifier = RandomForestClassifier(random_state = 23)
classifier.fit(X_train, y_train)

### Preprocessing for classification

relevantRockYouPasswords = parser.extractAllRelevantPasswords(rockYouDataframe, relevantDigraphDataframe)
unique_digraph_set = set(final_dataframe['digraph']) 

def _valueAtIndices(value, array):
    return [index for index, val in enumerate(y_test) if val==value]

def _getFeaturesAndLabelsForPassword(password, X_test, y_test):
    digraphArray = parser.getDigraphs(password)
    testFeaturesForPassword = []
    testLabels = []
    for digraph in digraphArray: 
        occurencesOfDigraph = _valueAtIndices(digraph, y_test)
        randomTestIndex = random.randint(occurencesOfDigraph[0], occurencesOfDigraph[-1])
        testFeaturesForPassword.append(X_test[randomTestIndex])
        testLabels.append(y_test[randomTestIndex])
    testFeaturesForPassword = np.array(testFeaturesForPassword)
    testLabels = np.array(testLabels)
    
    return testFeaturesForPassword, testLabels

def _getFeaturesAndLabelsOffseted(password, X_test, y_test, offset = 0.2):
    digraphArray = parser.getDigraphs(password)
    testFeaturesForPassword = []
    testLabels = []
    for digraph in digraphArray: 
        occurencesOfDigraph = _valueAtIndices(digraph, y_test)
        randomTestIndex = random.randint(occurencesOfDigraph[0], occurencesOfDigraph[-1])
        testFeaturesForPassword.append(X_test[randomTestIndex])
        testLabels.append(y_test[randomTestIndex])
    testFeaturesForPassword = np.array(testFeaturesForPassword)
    testLabels = np.array(testLabels)

def _get_top_probabilities(classifier, pred_prob_arr):
    return dict(sorted(dict(zip(classifier.classes_,pred_prob_arr)).items(), key=lambda x:x[1], reverse=True))
    
def get_top_digraphs(classifier,pred_prob_arr, no_of_digraphs=10):
    return list(_get_top_probabilities(classifier,pred_prob_arr).keys())[:no_of_digraphs]

def generate_words(arr_of_digraphs):
    pred_word = []
    for digraph_tuple in itertools.product(*arr_of_digraphs):
         pred_word.append(''.join(digraph_tuple))
    return pred_word

def calculatePenaltyScore(digraphProbabilites, testLabels):
    penaltyScore = 0
    diCount = 0
    for i in range(len(digraphProbabilites)): 
        row = digraphProbabilites[i]
        for j in range(len(row)):
            if row[j]==testLabels[i]:
                diCount+=1
                penaltyScore+= (j+1)
                break
    if diCount!=7:
        print("Error.")
    return penaltyScore

penaltyScores = {}
for password in relevantRockYouPasswords[:10]:
    testPassword = password
    testFeatures, testLabels = _getFeaturesAndLabelsForPassword(testPassword, X_test, y_test)    

    predictedProbabilites = classifier.predict_proba(testFeatures)
    digraph_prob=[]
    for row in predictedProbabilites:
        digraph_prob.append(get_top_digraphs(classifier,row, 307))
    
    penaltyScores[testPassword] = calculatePenaltyScore(digraph_prob, testLabels)
    index = originalRockYouDataframeWithCount.index[originalRockYouDataframeWithCount['password']== testPassword].tolist()[0]
    occurences = originalRockYouDataframeWithCount.iloc[index]['count']
    
    print(f"{testPassword} — Penalty:{calculatePenaltyScore(digraph_prob, testLabels)}", end=' ')
    print("Guess:",index, " Occurences:",occurences )


## Adding Threshold

#penaltyScoresWithOffset = {}
#for password in relevantRockYouPasswords:
#    testPassword = password
#    testFeatures, testLabels = _getFeaturesAndLabelsForPassword(testPassword, X_test, y_test)    
#
#    predictedProbabilites = classifier.predict_proba(testFeatures)
#    digraph_prob=[]
#    for row in predictedProbabilites:
#        digraph_prob.append(get_top_digraphs(classifier,row, 307))
#    
#    penaltyScores[testPassword] = calculatePenaltyScore(digraph_prob, testLabels)
    
#    print(f"{testPassword}: {calculatePenaltyScore(digraph_prob, testLabels)}")



#y_pred = classifier.predict(X_test)
#print ("Accuracy Score : {}%".format(accuracy_score(y_test, y_pred)*100))
#

