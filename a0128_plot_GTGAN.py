import numpy as np
# from helper import getNormalizedMatrix
# from numpy import linalg as LA
# import random
from networkx import nx
from matplotlib import pyplot as plt
from helper import getNormalizedMatrix_np
from helper import evaluate
from networkx import nx

# In[L2]


allSC = np.load('pred0128/raw_sc_test_RESTING_L2.npy')
trueFC = np.load('pred0128/raw_fc_test_RESTING_L2.npy')
predFC = np.load('pred0128/prediction_SC_RESTINGSTATE_partialcorrelationFC_L2_set3_lm1000.0.npy')[:,:,:,0]
# predFC = np.load('pred0128/raw_prediction_SC_RESTINGSTATE_partialcorrelationFC_L2_set3_lm1000.0.npy')[:,:,:,0]
pearson,diff = evaluate(predFC,trueFC,normalize=True)
print(np.mean(pearson),np.mean(diff))


# bestIdx = np.argmax(pearson)

minMSE = 100
for idx in range(allSC.shape[0]):
    sc = getNormalizedMatrix_np(allSC[idx])
    tfc = getNormalizedMatrix_np(trueFC[idx])
    pfc = getNormalizedMatrix_np(predFC[idx])
    # mse = np.linalg.norm(tfc-pfc)
    # # mse = np.sum(np.abs(tfc-pfc))
    # print(idx,mse)
    # if mse<minMSE:
    #     sc_best = sc
    #     tfc_best = tfc
    #     pfc_best = pfc
    #     idx_best = idx
    #     minMSE = mse
    if pearson[idx]>0.70:
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
        plt.savefig('case0128_GTGAN/rfmriL2_'+str(idx)+'.pdf')
        plt.show()



w = 3
h = 3

G = nx.from_numpy_array(allSC[0])
# pos = nx.spring_layout(G,scale=9)
# pos = nx.circular_layout(G)
pos = nx.fruchterman_reingold_layout(G)
# pos = nx.random_layout(G)
# pos = nx.spectral_layout(G)
# pos = nx.shell_layout(G)
# pos = nx.layout(G)





bestCount = 400
best_i = 0
th_best = -1
pos = nx.circular_layout(G)
# pos = nx.kamada_kawai_layout(G)
list = []
mul = 1
# mul = 2.5
# for i in listL2:
for i in range(allSC.shape[0]):
    for th in [0.05]:  # for L2 FC
    # for th in [0.03]:  # for full corr FC

        sc = getNormalizedMatrix_np(allSC[i])
        # sc = getNormalizedMatrix_np(allSC[i])
        sc[sc<th] = 0
        tfc = getNormalizedMatrix_np(np.copy(trueFC[i]))
        # tfc = np.copy(trueFC[i])
        # tfc[tfc!=0]=1
        tfc[np.abs(tfc)<th*mul] = 0

        # tfc[tfc>=th*mul] = 1
        # tfc[tfc <= -th * mul] = -1
        rawpfc = getNormalizedMatrix_np(np.copy(predFC[i]))

        pfc = np.copy(rawpfc)
        pfc[np.abs(pfc)<th]=0
        # pfc[pfc>=th] = 1
        # pfc[pfc<=-th]= -1
        # diff = np.count_nonzero(tfc-pfc)*1.0/max(np.count_nonzero(tfc),np.count_nonzero(pfc))

        # if(diff<30):
        # if pearson[i]>0.73:
        if pearson[i]>0.70:
        # if True:
            print(i,pearson[i])

            plt.figure(figsize=(16, 6))
            graph = nx.grid_2d_graph(1, 3)  # 4x4 grid

            # plt.figure(figsize=(w, h))
            plt.subplot(131,aspect=1.0)
            g = nx.from_numpy_array(sc)
            nx.draw(g, pos, node_color='g', node_size=100, font_size=10, edge_color='k', width=1, with_labels=True)
            # plt.title('Source graph: '+str(i))
            plt.title('Subject {}:  SC'.format(i))
            # plt.show()

            # plt.figure(figsize=(w, h))
            plt.subplot(132,aspect=1.0)
            g = nx.from_numpy_array(tfc)
            edges, weights = zip(*nx.get_edge_attributes(g, 'weight').items())
            # weights[weights>=0] = 1
            # weights[weights<=0] = -1
            nx.draw(g, pos, node_color='g', node_size=100, font_size=10, edge_color=weights, width=2, with_labels=True)
            # nx.draw(g, pos, node_color='g', node_size=100, font_size=10, edge_color='k', width=1, with_labels=True)
            plt.title('Subject {}: Empirical FC'.format(i))
            # plt.show()

            # plt.figure(figsize=(w, h))
            plt.subplot(133,aspect=1.0)
            g = nx.from_numpy_array(pfc)
            edges, weights = zip(*nx.get_edge_attributes(g, 'weight').items())
            # weights[weights >= 0] = 1
            # weights[weights <= 0] = -1
            nx.draw(g, pos, node_color='g', node_size=100, font_size=10, edge_color=weights, width=2,with_labels=True)

            # nx.draw(g, pos, node_color='g', node_size=100, font_size=10, edge_color='k', width=1, with_labels=True,title='Pred')
            # plt.title('Predicted target graph  diff={},  th = {:.4f}'.format(diff,th))
            plt.title('Subject {}: Predicted FC'.format(i))
            plt.savefig('case0128_GTGAN/net_rfmriL2_{}.pdf'.format(i))
            plt.show()

# In[full]



allSC = np.load('pred0128/raw_sc_test_RESTING.npy')
trueFC = np.load('pred0128/raw_fc_test_RESTING.npy')
predFC = np.load('pred0128/prediction_SC_RESTINGSTATE_correlationFC_set1_lm100000.0.npy')[:,:,:,0]

pearson,diff = evaluate(predFC,trueFC,normalize=True)
print(np.mean(pearson),np.mean(diff))


# bestIdx = np.argmax(pearson)

minMSE = 100
for idx in range(allSC.shape[0]):
    sc = getNormalizedMatrix_np(allSC[idx])
    tfc = getNormalizedMatrix_np(trueFC[idx])
    pfc = getNormalizedMatrix_np(predFC[idx])
    # mse = np.linalg.norm(tfc-pfc)
    # # mse = np.sum(np.abs(tfc-pfc))
    # print(idx,mse)
    # if mse<minMSE:
    #     sc_best = sc
    #     tfc_best = tfc
    #     pfc_best = pfc
    #     idx_best = idx
    #     minMSE = mse
    if pearson[idx]>0.70:
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
        plt.savefig('case0128_GTGAN/rfmriFull_'+str(idx)+'.pdf')
        # plt.savefig('case0128_GTGAN/rfmriL2_'+str(idx)+'.pdf')
        plt.show()



w = 3
h = 3

G = nx.from_numpy_array(allSC[0])
# pos = nx.spring_layout(G,scale=9)
# pos = nx.circular_layout(G)
pos = nx.fruchterman_reingold_layout(G)
# pos = nx.random_layout(G)
# pos = nx.spectral_layout(G)
# pos = nx.shell_layout(G)
# pos = nx.layout(G)





bestCount = 400
best_i = 0
th_best = -1
pos = nx.circular_layout(G)
# pos = nx.kamada_kawai_layout(G)
list = []
mul = 1
# mul = 2.5
# for i in listL2:
for i in range(allSC.shape[0]):
    # for th in [0.05]:  # for L2 FC
    for th in [0.02]:  # for full corr FC

        sc = getNormalizedMatrix_np(allSC[i])
        # sc = getNormalizedMatrix_np(allSC[i])
        sc[sc<th] = 0
        tfc = getNormalizedMatrix_np(np.copy(trueFC[i]))
        # tfc = np.copy(trueFC[i])
        # tfc[tfc!=0]=1
        tfc[np.abs(tfc)<th*mul] = 0

        # tfc[tfc>=th*mul] = 1
        # tfc[tfc <= -th * mul] = -1
        rawpfc = getNormalizedMatrix_np(np.copy(predFC[i]))

        pfc = np.copy(rawpfc)
        pfc[np.abs(pfc)<th]=0
        # pfc[pfc>=th] = 1
        # pfc[pfc<=-th]= -1
        # diff = np.count_nonzero(tfc-pfc)*1.0/max(np.count_nonzero(tfc),np.count_nonzero(pfc))

        # if(diff<30):
        if pearson[i]>0.7:
        # if True:
            print(i,pearson[i])

            plt.figure(figsize=(16, 6))
            graph = nx.grid_2d_graph(1, 3)  # 4x4 grid

            # plt.figure(figsize=(w, h))
            plt.subplot(131,aspect=1.0)
            g = nx.from_numpy_array(sc)
            nx.draw(g, pos, node_color='g', node_size=100, font_size=10, edge_color='k', width=1, with_labels=True)
            # plt.title('Source graph: '+str(i))
            plt.title('Subject {}:  SC'.format(i))
            # plt.show()

            # plt.figure(figsize=(w, h))
            plt.subplot(132,aspect=1.0)
            g = nx.from_numpy_array(tfc)
            edges, weights = zip(*nx.get_edge_attributes(g, 'weight').items())
            # weights[weights>=0] = 1
            # weights[weights<=0] = -1
            nx.draw(g, pos, node_color='g', node_size=100, font_size=10, edge_color=weights, width=2, with_labels=True)
            # nx.draw(g, pos, node_color='g', node_size=100, font_size=10, edge_color='k', width=1, with_labels=True)
            plt.title('Subject {}: Empirical FC'.format(i))
            # plt.show()

            # plt.figure(figsize=(w, h))
            plt.subplot(133,aspect=1.0)
            g = nx.from_numpy_array(pfc)
            edges, weights = zip(*nx.get_edge_attributes(g, 'weight').items())
            # weights[weights >= 0] = 1
            # weights[weights <= 0] = -1
            nx.draw(g, pos, node_color='g', node_size=100, font_size=10, edge_color=weights, width=2,with_labels=True)

            # nx.draw(g, pos, node_color='g', node_size=100, font_size=10, edge_color='k', width=1, with_labels=True,title='Pred')
            # plt.title('Predicted target graph  diff={},  th = {:.4f}'.format(diff,th))
            plt.title('Subject {}: Predicted FC'.format(i))
            plt.savefig('case0128_GTGAN/net_rfmriFull_{}.pdf'.format(i))
            plt.show()


# In[]

# allSC = np.load('tanh-results/rawSC_Normalized.npy')[691:]
# trueFC = np.load('tanh-results/rawFC_nonNormalized.npy')[691:]
# predFC = np.load('tanh-results/predFC_Resting_L2.npy')#[:,:,:,0]
#
# pearson,diff = evaluate(predFC,trueFC,normalize=True)
# print(np.mean(pearson),np.mean(diff))