import collections
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import NearMiss
from sklearn.preprocessing import StandardScaler
import RockYouDatasetParser 
import SystemPath
from multiprocess import Pool
import json 
import random

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

# relevantDigraphDataframe = pd.read_csv(path.getDataFilePath("uniqueDigraphs.csv"))
originalRockYouDataframeWithCount = pd.read_csv(path.getDataFilePath("rockyoudataset.csv"))

# Splitting the datasets for undersampling 
dataframe=pd.concat([ msuDataset ,greycWebDataset, greycDataset, stonyDataset])
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
X=finalDataframe.iloc[:,3:4].values
y=finalDataframe.iloc[:,0].values
_getDigraphFrequencies(finalDataframe)

# Scaling features
xTrain, xTest, yTrain, yTest = _trainTestSplit(finalDataframe, 0.30)
# scaler = StandardScaler()
# xTrain = scaler.fit_transform(xTrain)
xTestCopyForThreshold = xTest [:]
# xTest = scaler.transform(xTest)

#Classification
classifier = RandomForestClassifier(random_state = 23)
classifier.fit(xTrain, yTrain)

yPred = classifier.predict(xTest)
print ("Accuracy Score : {}%".format(accuracy_score(yTest, yPred)*100))

### Preprocessing for classification

relevantDigraphDataframe = pd.DataFrame(finalDataframe['digraph'])
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
    return dict(sorted(dict(zip(classifier.classes_,predictionProbabilityArray)).items(), key=lambda x:x[1], reverse=True))
    
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
                penaltyScore += (j+1)
                break
    if diCount!=7:
        print("Error. Digraph count = {diCount}")
    return penaltyScore


# Rank Penalties 
def rankPenalties(processData):
    testRuns, xTest, yTest, testPassword = processData
    bestGuesses = []
    print(testPassword)
    for i in range(testRuns): 
        print(i)
        penaltyScores = {}
        # 307 
        testFeatures, testLabels = _getFeaturesAndLabelsForPassword(testPassword, xTest, yTest)  
        predictedProbabilites = classifier.predict_proba(testFeatures)
        digraphProbabilities=[]
        for row in predictedProbabilites:
            digraphProbabilities.append(get_top_digraphs(classifier,row, 556))
            
        for password in relevantRockYouPasswords:
            curPasswordDigraph = []
            for i in range(1,len(password)):
                curPasswordDigraph.append(password[i-1:i+1])    
            penaltyScores[password] = calculatePenaltyScore(digraphProbabilities, curPasswordDigraph)
        
        intermediatePenaltyScoreDict = sorted(penaltyScores.items(), key=lambda kv: kv[1])
        sortedPenaltyScores = collections.OrderedDict(intermediatePenaltyScoreDict)
    
        bestGuesses.append(list(sortedPenaltyScores.keys()).index(testPassword))
    
    return bestGuesses  

def multiprocessingExperiment(thresholdValue):
      xTestWithOffset = []
      testPassword = "lamondre"
      # Update the xTest rows based on the threshold
      for row in xTestCopyForThreshold:
          newRow = row
          if abs(row[0]) < thresholdValue:
              newRow = [thresholdValue]
          xTestWithOffset.append(newRow)
          
      xTestWithOffset = np.array(xTestWithOffset)
      print(thresholdValue)
      poolData = ([50, xTest, yTest, testPassword], [50, xTestWithOffset, yTest, testPassword])
      
      with Pool(2) as processPool:
          res = processPool.map(rankPenalties, poolData)
         
      res.insert(0, thresholdValue)
      
      return res


# Test 

def rankPenaltiesRandomPasswords(poolData):
    testRuns, xTest, yTest, testPassword, isThreshold = poolData
    label = "WithThreshold" if isThreshold else "Without"
    randomGuess = relevantRockYouPasswords.index(testPassword)
    
    
    res = [testPassword, label, randomGuess]
    bestGuesses = []
    print(testPassword)
    for i in range(testRuns): 
        print(i)
        penaltyScores = {}
        # 307 
        testFeatures, testLabels = _getFeaturesAndLabelsForPassword(testPassword, xTest, yTest)  
        predictedProbabilites = classifier.predict_proba(testFeatures)
        digraphProbabilities=[]
        for row in predictedProbabilites:
            digraphProbabilities.append(get_top_digraphs(classifier,row, 556))
            
        for password in relevantRockYouPasswords:
            curPasswordDigraph = []
            for i in range(1,len(password)):
                curPasswordDigraph.append(password[i-1:i+1])    
            penaltyScores[password] = calculatePenaltyScore(digraphProbabilities, curPasswordDigraph)
        
        intermediatePenaltyScoreDict = sorted(penaltyScores.items(), key=lambda kv: kv[1])
        sortedPenaltyScores = collections.OrderedDict(intermediatePenaltyScoreDict)
    
        bestGuesses.append(list(sortedPenaltyScores.keys()).index(testPassword))
    
    res.append(bestGuesses[0])
    return res
    
def randomPasswordsExperiment(thresholdValue, testPasswords):
    xTestWithOffset = []
    for row in xTestCopyForThreshold:
          newRow = row
          if abs(row[0]) < thresholdValue:
              newRow = [thresholdValue]
          xTestWithOffset.append(newRow)
          
    xTestWithOffset = np.array(xTestWithOffset)
    
    # poolData = [testRuns, xTest, yTest, testPassword], [testRuns, xTestWithOffset, yTest, testPassword]
    poolData = []
    testRuns = 1
    for password in testPasswords:
        poolData.append([testRuns, xTest, yTest, password, False])
        poolData.append([testRuns, xTestWithOffset, yTest, password, True])
        
    
    with Pool(10) as processPool:
        res = processPool.map(rankPenaltiesRandomPasswords, poolData)
    
    res.insert(0, thresholdValue)
    return res 


thresholdValues = range(100, 350, 25)
n = 199
testPasswords = ["lamondre"] + random.sample(relevantRockYouPasswords, n)

# [testPassword, With/Without, randomGuess], bestGuess]
data = []
for threshold in thresholdValues: 
    data.append(randomPasswordsExperiment(threshold, testPasswords))

experiment = []
for dataRow in data: 
    threshold = dataRow[0]
    for row in dataRow[1:]:    
        row.append(threshold)
        experiment.append(row)
        
expData = pd.DataFrame(experiment, columns = ["Password", "Type", "Random Guess", "Guess", "Threshold"])
expData.to_csv('./Data/ExperimentData200Password.csv')

    


# data = []
# for thresholdValue in thresholdValues:
#     res = multiprocessingExperiment(thresholdValue)
#     data.append(res)

    
# experiment = []
# for row in data:
#     threshold, bestGuesses, bestGuessesWithOffset = row 
#     length = len(bestGuesses)
#     data = [[bestGuesses[i], bestGuessesWithOffset[i], threshold] for i in range(length)]
#     experiment.extend(data)


# experimentDataframe = pd.DataFrame(experiment, columns = ['withoutThreshold','withThreshold', 'thresholdValue'])

# experimentDataframe.to_csv('./Data/ExperimentData100-325.csv')
    
# relevantRockYouPasswords.index('lamondre')
