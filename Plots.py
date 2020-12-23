import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import collections 
import os 
import json 
import seaborn

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
plt.title("Average DDT Time (user-based) v/s Frequency")
plt.xlabel("Average DDT Time")
plt.ylabel("Frequency")
plt.hist(x = dataPoints, bins = getBins(dataPoints))



# SINGLE FEATURE - DDT 
# Accuracy: 3.37 % 


# Without: [1105, 14681, 15270, 2114, 110, 10592, 8945, 9, 43655, 80511]

# The thresholding suffers when the threshold is 100. I'm trying to see when it actually starts dropping

# Threshold: 100
# With: [1451, 1126, 6988, 409, 221, 133, 3702, 6767, 91, 1981]

# Threshold: 150
# With: [36186, 56717, 68347, 100029, 116324, 57706, 67388, 170580, 23029, 38070]

# Threshold: 200 
# With: [46209, 29631, 3854, 31924, 42443, 37789, 111042, 1450, 100183, 47740]

# Threshold: 250 
# With: [116986, 132027, 66862, 143520, 105982, 116986, 130783, 87243, 94642, 108524]

# Threshold: 300 
# With: [158240, 158240, 130356, 220544, 130356, 158240, 82598, 81326, 158240, 156784]


with open('./Data/experimentResults.txt', 'r') as inp:
    data = json.load(inp)
    
# Plot out change of threshold and avg of 10 best 
# Plot out change of threshold and 10 worst 
# Plot avg threshold vs random 









