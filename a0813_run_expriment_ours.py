# Created by qli10 at 8/13/2019
# In[]
from a0813run_model import *
import itertools
# In[]
trainSize = 700
rawData = np.loadtxt('original_sc_fc.txt', delimiter=',')
trainID = rawData[:trainSize, 0]
testID = rawData[:trainSize:, 0]

# In[]
boolList = [True,False]
combinations = list(itertools.product(*[boolList,boolList,boolList]))
res = []
i = 0
for setting in combinations:
    print ('setting ',i,setting)
    i = i+1
    useLaplacian, normalized, useAbs = setting
    pearsonCorrelation,diff = run_our_model(rawData,useLaplacian = useLaplacian,normalized = normalized,useAbs = useAbs,trainSize = trainSize)
    res.append([pearsonCorrelation.mean(),pearsonCorrelation.std()])
print(combinations)
print(res)
