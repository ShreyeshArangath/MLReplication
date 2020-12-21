import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import collections 
import os 

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
    extractedUserMap[userNo].append(interKey)


#### PLOTS

## PLOT: Inter Key Time v/s Frequency: {UID}
for userNo in extractedUserMap:
    interKey = extractedUserMap[userNo]
    plt.title("Inter Key Time v/s Frequency: {0}".format(userNo))
    plt.xlabel("Inter Key")
    plt.ylabel("Frequency")
    plt.hist(x = interKey, bins = getBins(interKey))
    break


## PLOT: Average Inter Key Time (user-based) v/s Frequency

averageUserData = {}
for userNo in extractedUserMap: 
    interKey = extractedUserMap[userNo]
    averageUserData[userNo] = np.mean(interKey)

dataPoints =  averageUserData.values() 
plt.title("Average Inter Key Time (user-based) v/s Frequency")
plt.xlabel("Average Inter Key Time")
plt.ylabel("Frequency")
plt.hist(x = dataPoints, bins = getBins(dataPoints))


# Threshold: 50 
# No Threshold: [3, 5262, 9424, 20695, 298, 3553, 8421, 178, 77139, 7873]
# With Threshold: [148730,148730,148730,148730,148730,148730,148730,148730,148730,148730]

# Threshold: 75
# No Threshold: [209, 5536, 268, 2566, 42727, 210, 6098, 61158, 2759, 196]
# With Threshold: [148728,148730,148730,148730,148730,148730,148726,148730,148730,148730]

# Threshold: 100 
# No Threshold: [2335, 8624, 1118, 189, 16215, 2230, 268, 20022, 0, 14223]
# With Threshold: [148730,148730,148726,148730,148730,148730,148730,148724,148730,148730]

    









