# Created by qli10 at 8/13/2019
# In[]
from a0813run_model import *
from a0814_comparison_functions import *
from a0814_baseline_function import *
import numpy as np
import itertools

# In[]
trainSize = 700
rawData = np.load('original_sc_fc_corr.npz')
# trainID = rawData[:trainSize, 0]
# testID = rawData[:trainSize:, 0]


# In[]
boolList = [True,False]
combinations = list(itertools.product(*[boolList,boolList,boolList]))
print('combinations:',combinations)
i = 0
means = np.zeros((len(combinations), 6))
stds = np.zeros((len(combinations), 6))
# In[]
for setting in combinations:
    print ('setting ',i,setting)

    useLaplacian, normalized, useAbs = setting
    try:
        corrOur,diff = run_our_model0814(rawData,useLaplacian = useLaplacian,
                                                normalized = normalized,
                                                useAbs = useAbs,
                                                trainSize = trainSize)
    except Exception as err:
        print("######## Our Throw Excepation: ",err)
        corrOur=0

    try:
        corr2014,corr2018 = paper20142018(rawData,useLaplacian = useLaplacian,normalized = normalized,useAbs = useAbs,trainSize = trainSize)
    except Exception as err:
        print("######## paper20142018 Throw Excepation: ", err,setting)
        corr2014 = 0
        corr2018 = 0


    try:
        corr2016 = paper2016(rawData,useLaplacian = useLaplacian,normalized = normalized,useAbs = useAbs,trainSize = trainSize)
    except Exception as err:
        print("######## paper2016 Throw Excepation: ", err,setting)
        corr2016 = 0


    try:
        corr2008 = paper2008(rawData,useLaplacian = useLaplacian,normalized = normalized,useAbs = useAbs,trainSize = trainSize)
    except Exception as err:
        print("######## paper2008 Throw Excepation: ", err,setting)
        corr2008 = 0

    try:
        corrBaseline = run_baseline(rawData, useLaplacian=useLaplacian, normalized=normalized, useAbs=useAbs,
                                trainSize=trainSize)
    except Exception as err:
        print("######## Baseline Throw Excepation: ", err,setting)
        corrBaseline = 0
    means[i, :] = np.asarray([np.mean(corr2008), np.mean(corr2014), np.mean(corr2016), np.mean(corr2018), np.mean(corrBaseline), np.mean(corrOur)])
    stds[i, :] = np.asarray([np.std(corr2008), np.std(corr2014), np.std(corr2016), np.std(corr2018), np.std(corrBaseline), np.std(corrOur)])
    i = i + 1
    # res.append([pearsonCorrelation.mean(),pearsonCorrelation.std()])
print(combinations)
print(means)
# np.savez('result0814_corrFC', settings=combinations, means=means,stds=stds)
# print(res)
