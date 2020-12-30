import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import collections 
import os 
import json 
import seaborn as sns

"""
Components: 
    1. Histograms based on inter-key times for the user  
    2. User's w/ distinctly different key times 
    

all users and one user 
    freq v/s inter-key

Share results using threshold 
    a. Table w/ threshold 
    b. Figure thresholds (thresholds v/s value(average))
    
Data decomposition 
    UserNumber   Inter-Key 

userNumber: [ik_0, ik_1, ik_2, ...., ik_99]
    
"""

### PREPROCESSING 
def getBins(data):
    n = len(data)
    maxData, minData = max(data), min(data)
    dataRange = maxData - minData
    noOfIntervals = int( n**(0.5) )
    widthOfIntervals = dataRange//noOfIntervals
    i = (minData//10)*10
    res = [i]
    while i <= (maxData//10 + 1)*10:
        i += widthOfIntervals
        res.append(i)
    return res 
    
dataframe = pd.read_csv('./Data/userData.csv')
del dataframe['Unnamed: 0']

extractedUserMap = collections.defaultdict(list)

for index, row in dataframe.iterrows():
    digraph, interKey, uut, ddt, userNo = row     
    extractedUserMap[userNo].append(ddt)


#### PLOTS

## PLOT: Inter Key Time v/s Frequency: {UID}
for userNo in extractedUserMap:
    interKey = extractedUserMap[20]
    plt.title("DDT v/s Frequency: {0}".format(userNo))
    plt.xlabel("DDT")
    plt.ylabel("Frequency")
    plt.hist(x = interKey, bins = getBins(interKey))
    break


## PLOT: Average Inter Key Time (user-based) v/s Frequency

averageUserData = {}
for userNo in extractedUserMap: 
    interKey = extractedUserMap[userNo]
    averageUserData[userNo] = np.mean(interKey)




dataPoints =  np.array( list(averageUserData.values()) )
dataPoints = dataPoints[np.where(dataPoints < 1000)]
plt.title("Average DDT Time (user-ba sed) v/s Frequency")
plt.xlabel("Average DDT Time")
plt.ylabel("Frequency")
plt.hist(x = dataPoints, bins = getBins(dataPoints))



data = pd.read_csv('./Data/ExperimentData100-325.csv')
# Plot out change of threshold and avg of 10 best 
# Plot out change of threshold and 10 worst 
# Plot avg threshold vs random 


# comparing the avg thresholds at each thresholdValue
averageThreshold = data.groupby('thresholdValue', as_index = False).agg(averageGuess = ('withThreshold', 'mean'))
ax = sns.catplot(x = 'thresholdValue', y = 'averageGuess', kind = 'bar', data = averageThreshold)
ax.set(xlabel = 'Threshold Values', ylabel = 'Number of Guesses', title = 'Average guess: With Threshold ')
plt.show()


# Comparing worst thresholds at each thresholdValue
worstThreshold = data.groupby('thresholdValue', as_index = False).agg(worstGuess = ('withThreshold', 'max'))
ax = sns.catplot(x = 'thresholdValue', y = 'worstGuess', kind = 'bar', data = worstThreshold)
ax.set(xlabel = 'Threshold Values', ylabel = 'Number of Guesses', title = 'Worst guess: With Threshold')
plt.show()


# Comparing avg of both with and without threshold experiments for each threshold value
avgComparison = data.groupby('thresholdValue', as_index = False).agg(thresholdAvgGuess = ('withThreshold', 'mean'), 
                                                                     modelAvgGuess = ('withoutThreshold', 'mean'))
avgComparison = pd.melt(avgComparison, id_vars = 'thresholdValue', var_name = 'variable', value_name = 'value')
ax = sns.catplot(x = 'thresholdValue', y = 'value', hue ='variable',  kind = 'bar', data = avgComparison)
ax.set(xlabel = 'Threshold Values', ylabel = 'Number of Guesses', title = 'Average guess: With Threshold v/s Without Threshold')
plt.show()


# Comparing worst of both with and without threshold experiments for each threshold value
worstComparison = data.groupby('thresholdValue', as_index = False).agg(withThreshold = ('withThreshold', 'max'), 
                                                                       withoutThreshold = ('withoutThreshold', 'max'))
worstComparison = pd.melt(worstComparison, id_vars = 'thresholdValue', var_name = 'variable', value_name = 'value')
ax = sns.catplot(x = 'thresholdValue', y = 'value', hue ='variable',  kind = 'bar', data = worstComparison)
ax.set(xlabel = 'Threshold Values', ylabel = 'Number of Guesses', title = 'Worst guess: With Threshold v/s Without Threshold')
plt.show()


# Comparing avg of both with and without threshold experiments for each threshold value along with random guess
avgComparison = data.groupby('thresholdValue', as_index = False).agg(
    thresholdAvgGuess = ('withThreshold', 'mean'), modelAvgGuess = ('withoutThreshold', 'mean'))
avgComparison['Random Guess'] = [55110]*len(avgComparison)
avgComparison = pd.melt(avgComparison, id_vars = 'thresholdValue', var_name = 'variable', value_name = 'value')
ax = sns.catplot(x = 'thresholdValue', y = 'value', hue ='variable',  kind = 'bar', data = avgComparison)
ax.set(xlabel = 'Threshold Values', ylabel = 'Number of Guesses', title = 'Average guess: With Threshold v/s Without Threshold  v/s Random Guessing')
plt.show()


# Comparing worst of both with and without threshold experiments for each threshold value along with random guess
worstComparison = data.groupby('thresholdValue', as_index = False).agg(
    withThreshold = ('withThreshold', 'max'), withoutThreshold = ('withoutThreshold', 'max'))
worstComparison['Random Guess'] = [55110]*len(worstComparison)
worstComparison = pd.melt(worstComparison, id_vars = 'thresholdValue', var_name = 'variable', value_name = 'value')
ax = sns.catplot(x = 'thresholdValue', y = 'value', hue ='variable',  kind = 'bar', data = worstComparison)
ax.set(xlabel = 'Threshold Values', ylabel = 'Number of Guesses', title = 'Worst guess: With Threshold v/s Without Threshold v/s Random Guessing')
plt.show()




# Comparing overall model guessing capabilities 
ax = sns.boxplot(x = data['withoutThreshold'], palette = 'Set3', width= 0.3)
ax.set(xlabel = 'Number of Guesses', title = 'Model prediction without threshold')
plt.show()







