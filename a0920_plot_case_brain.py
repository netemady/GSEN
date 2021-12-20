# In[]
import numpy as np
# from helper import getNormalizedMatrix
# from numpy import linalg as LA
# import random
from networkx import nx
from matplotlib import pyplot as plt
from helper import getNormalizedMatrix_np
w = 3
h = 3
# result = np.load('case_real_IOT20.npz')
result = np.load('case_SC_RESTINGSTATE_correlationFC.npz')
# result = np.load('case_SC_RESTINGSTATE_correlationFC_best.npz')
# result = np.load('case_SC_RESTINGSTATE_correlationFC_mse.npz')
allSC = -result['testSC']
diagIndices = np.arange(0,allSC.shape[1])
trueFC = -result['fc_test']
predFC = -result['predFC']
pearson = result['res']
bestIdx = np.argmax(pearson)
# predFC[predFC<0]=0

# trueFC = np.abs(result['fc_test'])
# predFC = np.abs(result['predFC'])

# trueFC[trueFC<0]=0
# predFC[predFC<0]=0
minMSE = 100
# for idx in [bestIdx]:
for idx in range(allSC.shape[0]):
    sc = getNormalizedMatrix_np(allSC[idx])
    tfc = getNormalizedMatrix_np(trueFC[idx])
    pfc = getNormalizedMatrix_np(predFC[idx])
    # sc = np.log(sc+1)
    # tfc[tfc<0.001]=0
    # pfc[pfc<0.001]=0
    # sc[sc > 0.01] = sc[sc > 0.01] * 0.05
    # tfc[tfc < 0.01] = 0
    # pfc[pfc < 0.01] = 0
    # tfc[tfc < 0] = 0
    # pfc[pfc < 0] = 0
    mse = np.linalg.norm(tfc-pfc)
    # mse = np.sum(np.abs(tfc-pfc))
    print(idx,mse)
    if mse<minMSE:
        sc_best = sc
        tfc_best = tfc
        pfc_best = pfc
        idx_best = idx
        minMSE = mse
    plt.figure(figsize=(9, 3))
    plt.subplot(131)
    plt.xlabel('Brain ROI index')
    plt.ylabel('Brain ROI index')
    plt.title('SC')
    plt.imshow(sc, cmap='hot', interpolation='nearest')
    # plt.colorbar()
    plt.subplot(132)
    plt.imshow(tfc, cmap='hot', interpolation='nearest')
    plt.xlabel('Brain ROI index')
    # plt.ylabel('Brain ROI index')
    plt.title('Empirical FC')
    # plt.colorbar()
    plt.subplot(133)
    plt.imshow(pfc, cmap='hot', interpolation='nearest')
    plt.xlabel('Brain ROI index')
    # plt.ylabel('Brain ROI index')
    plt.title('Predicted FC')
    # plt.colorbar()
    # plt.title(str(idx))
    plt.savefig('case_SC_FC/case_brain_rfmri_'+str(idx)+'.pdf')
    plt.show()

# plt.figure(figsize=(20, 15))
# plt.subplot(131)
# plt.imshow(sc_best, cmap='hot', interpolation='nearest')
# plt.subplot(132)
# plt.imshow(tfc_best, cmap='hot', interpolation='nearest')
# plt.subplot(133)
# plt.imshow(pfc_best, cmap='hot', interpolation='nearest')
# plt.show()
#
# idx_visual_best = 121
# idx_visual_best = 82
# # idx_visual_best = 81
# # idx_visual_best = 41
# sc = getNormalizedMatrix_np(allSC[idx_visual_best])
# tfc = getNormalizedMatrix_np(trueFC[idx_visual_best])
# pfc = getNormalizedMatrix_np(predFC[idx_visual_best])
# # sc[sc>0.01]=sc[sc>0.01]*0.05
# # th = 0.03
# # sc[sc>th]=th
# sc = np.log(sc)
#
# #
# # tfc[tfc<0.001]=0
# # pfc[pfc<0.001]=0
# # tfc = np.log(tfc)
# # pfc = np.log(pfc)
# plt.figure(figsize=(12,5))
# plt.subplot(131)
# plt.imshow(sc, cmap='hot', interpolation='nearest')
# plt.xlabel('Brain ROI index')
# plt.ylabel('Brain ROI index')
# plt.title('Structural Connectivity')
# plt.subplot(132)
# plt.imshow(tfc, cmap='hot', interpolation='nearest')
# plt.xlabel('Brain ROI index')
# plt.ylabel('Brain ROI index')
# plt.title('Empirical Functional Connectivity')
# plt.subplot(133)
# plt.imshow(pfc, cmap='hot', interpolation='nearest')
# plt.xlabel('Brain ROI index')
# plt.ylabel('Brain ROI index')
# plt.title('Predicted Functional Connectivity')
#
# plt.savefig('case_brain_rfmri.pdf')
# plt.show()
# np.savez('case_brain_rfmri.npz',idx=idx_visual_best,sc=sc,tfc=tfc,pfc=pfc)

