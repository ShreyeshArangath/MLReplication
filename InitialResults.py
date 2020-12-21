

import numpy as np 
from scipy.stats import norm 
import matplotlib.pyplot as plt
import math
import seaborn as sns


sns.set_theme(style="whitegrid")

data =[16152, 1, 865, 30411, 101104, 403, 357, 177,
      2162, 461, 21956, 24496, 248, 7105, 2711, 27528, 3468, 161, 75987, 1996, 3773, 10072, 
      31332, 21460, 296, 346, 1315, 1583, 630, 1055, 2594, 12042, 242, 645, 48, 77112, 0, 1054,
      2937, 3172, 2584, 0, 689, 45280, 2138, 545, 2508, 6504, 182, 438, 228, 103, 87478, 258, 4788,
      223, 6680, 1598, 4050, 21, 20543, 370, 48373, 31863, 176, 7847, 3742, 78463, 14926, 601, 13422, 
      3881, 68189, 165, 639, 199, 1127, 22492, 2175, 13188, 77734, 30938, 4358, 2328, 294, 765, 3, 777,
      496, 26892, 93836, 717, 167, 9849, 1013, 19868, 16768, 190, 10644, 151601]



avg = np.average(data)
var = (np.var(data))

sns.boxplot(x=data, width = 0.1)
sns.scatterplot(y=data, x=range(100))

op = np.array(data)
print(len(op[op<=55110]))

print(len(data))
print(avg, var**0.5, np.median(data))

print(397213/6514177*100)
print(55110/269230*100)
print(301906/6514177*100)
print(14373.71/269230*100)
print(min(data))


# 4 components  
#   1. Histograms based on the interkey times from each user  (freq v/s inter-key time)
#   2. Users with distinctly different key times (box plot can help save that)
#   3. Share results using threshold
#       Table with threshold (thresholds v/s average value)
#       Figure out thresholds 






