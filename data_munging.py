import pandas
import numpy

data = pandas.read_csv("ExperimentData1.csv")

print(data)
print(data.shape)

withThresholdRes = dict()
withoutThresholdRes = dict()
counter = 0

for x in range(data.shape[0]):
    if data["thresholdValue"][x] in withThresholdRes:
        withThresholdRes[data["thresholdValue"][x]] += data["withThreshold"][x]
        withoutThresholdRes[data["thresholdValue"][x]] += data["withoutThreshold"][x]
        counter += 1
    else:
        withThresholdRes[data["thresholdValue"][x]] = data["withThreshold"][x]
        withoutThresholdRes[data["thresholdValue"][x]] = data["withoutThreshold"][x]
        print(counter)
        counter = 1

withRes = []
for key in withThresholdRes.keys():
    withThresholdRes[key] = withThresholdRes[key] / counter
    withRes.append(withThresholdRes[key])
    withoutThresholdRes[key] = withoutThresholdRes[key] / counter
    withRes.append(withoutThresholdRes[key])

print(withThresholdRes)
print(withoutThresholdRes)
print(withRes)

ddt_data = pandas.read_csv("userData.csv")

print(ddt_data["ddt"])
hist, bin_edges = numpy.histogram(ddt_data["ddt"], [x*25 for x in range(21)])

vals = []
for x in range(len(hist)):
    vals.append(hist[x] / float(sum(hist)))

print(vals)

print(bin_edges)

# [14611.5 17279.92; 21964.64 19978.8; 23434.16 25514.9; 37865.62 18297.24; 32674.7 27257.5; 64033.9 19778.58; 49425.84 21557.44; 67305.14 18510.8; 78259.98 23615.84; 116847.8 17450.32; 88733.08 12657.48; 163088.5 15287.88]
