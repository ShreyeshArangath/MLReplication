import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score
from imblearn.under_sampling import NearMiss
from sklearn.preprocessing import StandardScaler
import RockYouDatasetParser 
import SystemPath

path = SystemPath.Path()
parser = RockYouDatasetParser.RockYouDatasetParser()

def _getDigraphFrequencies(dataframe):
    """Returns the digraph frequencies within a dataframe."""
    return dataframe.groupby("digraph").count()

""" OBSOLETE """
def _perfectTestSplit(dataframe, testSize=0.2):
    """Returns a train dataframe and a perfectly split test dataframe."""
    uniqueDigraphSet = set(dataframe['digraph']) 
    testSplit = 0.2
    minOccurencesForPerfectSplit = int(len(dataframe)*testSplit//len(uniqueDigraphSet))
    trainDataframe = pd.DataFrame()
    testDataframe = pd.DataFrame()
    
    for digraph in uniqueDigraphSet: 
        temp_dataframe = dataframe[dataframe['digraph']==digraph]
        temp_dataframe = temp_dataframe.sample(frac=1).reset_index(drop=True)
        temp_testDataframe = temp_dataframe.iloc[:minOccurencesForPerfectSplit,:]
        temp_trainDataframe = temp_dataframe.iloc[minOccurencesForPerfectSplit:,:]
        trainDataframe = pd.concat([trainDataframe, temp_trainDataframe], ignore_index=True, sort=False)
        testDataframe = pd.concat([testDataframe, temp_testDataframe], ignore_index=True, sort=False)
        
    return trainDataframe, testDataframe

def _getBinCount(numpyArray):
    classes, indices = np.unique(numpyArray, return_inverse=True)
    class_counts = np.bincount(indices)
    return class_counts

def _trainTestSplit(finalDataframe, testSize=0.2):
    """Returns xTrain, xTest, yTrain, yTest. The test data returned is perfectly balanced."""
    
    uniqueDigraphSet = set(finalDataframe['digraph']) 
    xTrain, xTest , yTrain , yTest = train_test_split(X, y, test_size=testSize, random_state = 0)
    min_instance_count = np.min(_getBinCount(yTest))
    print("The minimum number of instances required for a perfectly balanced test data: ",min_instance_count)
    
    yTestBalanced = []
    yTestDiscard = []
    xTestBalanced = []
    xTestDiscard = []
    indices_to_discard = []   
    for digraph in uniqueDigraphSet: 
        indices = (np.where(yTest==digraph))[0]
        indices_to_keep = indices[:np.min(min_instance_count)]
        indices_to_discard = indices[np.min(min_instance_count):]
        yTestBalanced.extend(np.array(yTest)[indices_to_keep])
        yTestDiscard.extend(np.array(yTest)[indices_to_discard])
        xTestBalanced.extend(np.array(xTest)[indices_to_keep])
        xTestDiscard.extend(np.array(xTest)[indices_to_discard])
    
    xTestDiscard= np.array(xTestDiscard)
    xTestBalanced = np.array(xTestBalanced)
    yTestBalanced = np.array(yTestBalanced)
    yTestDiscard = np.array(yTestDiscard)
    
    xTrainUpdated = np.concatenate((xTrain, xTestDiscard))
    yTrainUpdated = np.concatenate((yTrain, yTestDiscard))
    
    print("The instances per digraph in testing data: ", _getBinCount(yTestBalanced))
    print("The instances per digraph in training data: ", _getBinCount(yTrainUpdated))
    
    return xTrainUpdated, xTestBalanced, yTrainUpdated, yTestBalanced

    
# Importing datasets
msuDataset=pd.read_csv(path.getDataFilePath("msuupdated.csv"))
stonyDataset = pd.read_csv(path.getDataFilePath("stonybrooksdataset_updated.csv"))
greycWebDataset = pd.read_csv(path.getDataFilePath("greycwebdata.csv"))
greycDataset = pd.read_csv(path.getDataFilePath("greyc_normal.csv"))
rockYouDataframe = pd.read_csv(path.getDataFilePath("rockyou8subset.csv"))
relevantDigraphDataframe = pd.read_csv(path.getDataFilePath("uniqueDigraphs.csv"))
originalRockYouDataframeWithCount = pd.read_csv(path.getDataFilePath("rockyoudataset.csv"))

# Splitting the datasets for undersampling 
dataframe=pd.concat([ msuDataset ,greycWebDataset, greycDataset])
dataframe=dataframe.groupby("digraph").filter(lambda x: len(x) > 100)
dataframeWithLessThan1000Samples = dataframe.groupby("digraph").filter(lambda x: len(x) < 1000) 
dataframeToUnderSample = dataframe.groupby("digraph").filter(lambda x: len(x) >= 1000)

# Under Sampling 
X=dataframeToUnderSample.iloc[:,2:].values
y=dataframeToUnderSample.iloc[:,1].values
undersample = NearMiss()
xUnder, yUnder = undersample.fit_resample(X, y)
# To delete the extra index column 
del dataframeWithLessThan1000Samples['Unnamed: 0'] 
undersampledDataframe = pd.DataFrame(xUnder, columns= ['inter-key','uut', 'ddt'])
undersampledDataframe.insert(0, "digraph", yUnder)

# Preprocessing 
finalDataframe = pd.concat([undersampledDataframe, dataframeWithLessThan1000Samples])
X=finalDataframe.iloc[:,1:].values
y=finalDataframe.iloc[:,0].values
_getDigraphFrequencies(finalDataframe)

# Scaling features
xTrain, xTest, yTrain, yTest = _trainTestSplit(finalDataframe, 0.30)
scaler = StandardScaler()
xTrain = scaler.fit_transform(xTrain)
xTestCopyForThreshold = xTest [:]
xTest = scaler.transform(xTest)

#Classification
classifier = RandomForestClassifier(random_state = 23)
classifier.fit(xTrain, yTrain)

#yPred = classifier.predict(xTest)
#print ("Accuracy Score : {}%".format(accuracy_score(yTest, yPred)*100))


### Preprocessing for classification

relevantRockYouPasswords = parser.extractAllRelevantPasswords(rockYouDataframe, relevantDigraphDataframe)
uniqueDigraphSet = set(finalDataframe['digraph']) 

def _valueAtIndices(value, array):
    """ Returns a list of indices based on a given list of indices """
    return [index for index, val in enumerate(array) if val==value]

def _getFeaturesAndLabelsForPassword(password, xTest, yTest):
    """ Returns the test features and test labels for a given password """
    digraphArray = parser.getDigraphs(password)
    testFeaturesForPassword = []
    testLabels = []
    for digraph in digraphArray: 
        occurencesOfDigraph = _valueAtIndices(digraph, yTest)
        randomTestIndex = random.randint(occurencesOfDigraph[0], occurencesOfDigraph[-1])
        testFeaturesForPassword.append(xTest[randomTestIndex])
        testLabels.append(yTest[randomTestIndex])
    testFeaturesForPassword = np.array(testFeaturesForPassword)
    testLabels = np.array(testLabels)
    
    return testFeaturesForPassword, testLabels

def _getFeaturesAndLabelsOffseted(password, xTest, yTest, offset = 0.2):
    """ Returns the offseted test features and test labels for a given password """
    digraphArray = parser.getDigraphs(password)
    testFeaturesForPassword = []
    testLabels = []
    for digraph in digraphArray: 
        occurencesOfDigraph = _valueAtIndices(digraph, yTest)
        randomTestIndex = random.randint(occurencesOfDigraph[0], occurencesOfDigraph[-1])
        testFeaturesForPassword.append(xTest[randomTestIndex])
        testLabels.append(yTest[randomTestIndex])
    testFeaturesForPassword = np.array(testFeaturesForPassword)
    testLabels = np.array(testLabels)

def _getTopProbabilities(classifier, predictionProbabilityArray):
    return dict(sorted(dict(zip(classifier.classes_,predictionProbabilityArray)).items(), 
                       key=lambda x:x[1], reverse=True))
    
def get_top_digraphs(classifier,predictionProbabilityArray, numberOfDigraphsToPredict=10):
    return list(_getTopProbabilities(classifier,predictionProbabilityArray).keys())[:numberOfDigraphsToPredict]

def calculatePenaltyScore(digraphProbabilites, testLabels):
    """ Calculate the penalty score for a password given a list of digraph probabilities """
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
loopIndex = 0
for password in relevantRockYouPasswords[:10]:
    testPassword = password
    testFeatures, testLabels = _getFeaturesAndLabelsForPassword(testPassword, xTest, yTest)    

    predictedProbabilites = classifier.predict_proba(testFeatures)
    digraphProbabilities=[]
    for row in predictedProbabilites:
        digraphProbabilities.append(get_top_digraphs(classifier,row, 307))
    
    penaltyScores[testPassword] = calculatePenaltyScore(digraphProbabilities, testLabels)
    index = originalRockYouDataframeWithCount.index[originalRockYouDataframeWithCount['password']== testPassword].tolist()[0]
    occurences = originalRockYouDataframeWithCount.iloc[index]['count']
    loopIndex+=1
    
    print(f"{loopIndex}. {testPassword} — Penalty:{calculatePenaltyScore(digraphProbabilities, testLabels)}", end=' ')
    print("Guess:",index, " Occurences:",occurences )
    
    
# Adding Threshold
print("\n\n Theshold EXPERIMENT \n\n")
penaltyScoresWithOffset = {}
thresholdValue = 50
xTestWithOffset = []
for row in xTestCopyForThreshold:
    row = np.array(row)
    row[row<thresholdValue] = thresholdValue
    xTestWithOffset.append(row)

xTestWithOffset = np.array(xTestWithOffset)

    
loopIndex = 0
for password in relevantRockYouPasswords[:10]:
    testPassword = password
    testFeatures, testLabels = _getFeaturesAndLabelsForPassword(testPassword, xTestWithOffset, yTest)    

    predictedProbabilites = classifier.predict_proba(testFeatures)
    digraphProbabilities=[]
    for row in predictedProbabilites:
        digraphProbabilities.append(get_top_digraphs(classifier,row, 307))
    
    penaltyScores[testPassword] = calculatePenaltyScore(digraphProbabilities, testLabels)
    index = originalRockYouDataframeWithCount.index[originalRockYouDataframeWithCount['password']== testPassword].tolist()[0]
    occurences = originalRockYouDataframeWithCount.iloc[index]['count']
    loopIndex+=1
    
    print(f"{loopIndex}. {testPassword} — Penalty:{calculatePenaltyScore(digraphProbabilities, testLabels)}", end=' ')
    print("Guess:",index, " Occurences:",occurences )
    

# y_pred = classifier.predict(xTest)
# print ("Accuracy Score : {}%".format(accuracy_score(yTest, y_pred)*100))


