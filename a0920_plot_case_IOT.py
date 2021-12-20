# In[]
import numpy as np
# from helper import getNormalizedMatrix
# from numpy import linalg as LA
# import random
from networkx import nx
from matplotlib import pyplot as plt

w = 3
h = 3
# result = np.load('case_real_IOT20.npz')
# result = np.load('case_real_IOT20_beta1.npz')
result = np.load('case_real_IOT40.npz')
allSC = np.abs(result['testSC'])
trueFC = -result['fc_test']
predFC = -result['predFC']
predFC[predFC<0]=0

# trueFC = np.abs(result['fc_test'])
# predFC = np.abs(result['predFC'])

# trueFC[trueFC<0]=0
predFC[predFC<0]=0
for idx in range(allSC.shape[0]):
    allSC[idx] = (allSC[idx]+np.transpose(allSC[idx]))*0.5
    trueFC[idx] = (trueFC[idx]+np.transpose(trueFC[idx]))*0.5
    predFC[idx] = (predFC[idx]+np.transpose(predFC[idx]))*0.5
pearson = result['res']
G = nx.from_numpy_array(allSC[1])
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
# pos = nx.circular_layout(G)
# pos = nx.kamada_kawai_layout(G)
list = []
# mul = 1.85
# mul = 1.7  # best for IOT-20
mul = 2.5
for i in range(allSC.shape[0]):
# for i in [35]:
# for i in [1]:
#     bestCount = 500
#     th_best = -1
#     if(pearson[i]<0.9):
#         continue
#     for th in np.arange(0.015,np.max(trueFC[i]),0.001):
    for th in np.arange(0.02,0.035,0.001):
        tfc = np.copy(trueFC[i])
        # tfc[tfc!=0]=1
        tfc[tfc<th*mul] = 0
        tfc[tfc>=th*mul] = 1
        pfc = np.copy(predFC[i])
        pfc[pfc<th]=0
        pfc[pfc>=th]=1
        diff = np.count_nonzero(tfc-pfc)
        if diff<bestCount:
            bestCount = diff
            th_best = th
            best_i = i
        # if diff<30:
        if diff<40:
        # if True:
            tfc = np.copy(trueFC[i])
            tfc[tfc < th*mul] = 0
            tfc[tfc >= th*mul] = 1
            pfc = np.copy(predFC[i])
            pfc[pfc < th] = 0
            pfc[pfc >= th] = 1

            plt.figure(figsize=(10, 3))
            graph = nx.grid_2d_graph(1, 3)  # 4x4 grid

            # plt.figure(figsize=(w, h))
            plt.subplot(131)
            g = nx.from_numpy_array(allSC[i])
            nx.draw(g, pos, node_color='g', node_size=100, font_size=10, edge_color='k', width=0.25, with_labels=True)
            # plt.title('Source graph: '+str(i))
            plt.title('Source graph')
            # plt.show()

            # plt.figure(figsize=(w, h))
            plt.subplot(132)
            g = nx.from_numpy_array(tfc)
            nx.draw(g, pos, node_color='g', node_size=100, font_size=10, edge_color='k', width=0.25, with_labels=True)
            plt.title('Empirical target graph:')
            # plt.show()

            # plt.figure(figsize=(w, h))
            plt.subplot(133)
            g = nx.from_numpy_array(pfc)
            nx.draw(g, pos, node_color='g', node_size=100, font_size=10, edge_color='k', width=0.25, with_labels=True,
                    title='Pred')
            # plt.title('Predicted target graph  diff={},  th = {:.4f}'.format(diff,th))
            plt.title('Predicted target graph')
            plt.savefig('case_IOT/IoT_40_case_{}_diff{}.pdf'.format(i,diff))
            plt.show()

    # print('current bestCount', bestCount)
    # print('current th_best', th_best)
    list.append(bestCount)
    # if bestCount>51:
    #     continue
    # else:
    #     print('bestCount', bestCount)
    #     print('th_best', th_best)

    #
    # if pearson[i]<0.98:
    #     continue
    # num_zeros = np.count_nonzero(allSC[i])
    # # num_zeros = np.count_nonzero(trueFC[i])
    # print('num_nonzeros:',i, num_zeros)

    # if num_zeros>200:
    #     continue

    # nx.draw_spectral(sc)
    # plt.show()

# best = np.asarray(list)


# tfc = np.copy(trueFC[best_i])
# tfc[tfc < th_best] = 0
# tfc[tfc >= th_best] = 1
# pfc = np.copy(predFC[best_i])
# pfc[pfc < th_best] = 0
# pfc[pfc >= th_best] = 1
#
# plt.figure(figsize=(10, 3))
# graph=nx.grid_2d_graph(1,3)  #4x4 grid
#
# # plt.figure(figsize=(w, h))
# plt.subplot(131)
# g = nx.from_numpy_array(allSC[best_i])
# nx.draw(g, pos, node_color='g', node_size=100, font_size=10, edge_color='k', width=0.25, with_labels=True)
# plt.title('Source graph')
# # plt.show()
#
# # plt.figure(figsize=(w, h))
# plt.subplot(132)
# g = nx.from_numpy_array(tfc)
# nx.draw(g, pos, node_color='g', node_size=100, font_size=10, edge_color='k', width=0.25, with_labels=True)
# plt.title('Empirical target graph')
# # plt.show()
#
# # plt.figure(figsize=(w, h))
# plt.subplot(133)
# g = nx.from_numpy_array(pfc)
# nx.draw(g, pos, node_color='g', node_size=100, font_size=10, edge_color='k', width=0.25, with_labels=True, title='Pred')
# plt.title('Predicted target graph')
# plt.show()

th = th_best
i = best_i

tfc = np.copy(trueFC[i])
tfc[tfc < th*mul] = 0
tfc[tfc >= th*mul] = 1
pfc = np.copy(predFC[i])
pfc[pfc < th] = 0
pfc[pfc >= th] = 1





plt.figure(figsize=(10, 3))
graph = nx.grid_2d_graph(1, 3)  # 4x4 grid

# plt.figure(figsize=(w, h))
plt.subplot(131)
g = nx.from_numpy_array(allSC[i])
nx.draw(g, pos, node_color='g', node_size=100, font_size=10, edge_color='k', width=0.25, with_labels=True)
plt.title('Source graph')
# plt.show()

# plt.figure(figsize=(w, h))
plt.subplot(132)
g = nx.from_numpy_array(tfc)
nx.draw(g, pos, node_color='g', node_size=100, font_size=10, edge_color='k', width=0.25, with_labels=True)
plt.title('Empirical target graph')
# plt.show()

# plt.figure(figsize=(w, h))
plt.subplot(133)
g = nx.from_numpy_array(pfc)
nx.draw(g, pos, node_color='g', node_size=100, font_size=10, edge_color='k', width=0.25, with_labels=True,
        title='Pred')
plt.title('Predicted target graph diff={},  th = {:.4f}'.format(bestCount,th_best))
# plt.savefig('IoT_case.pdf')
plt.show()






plt.figure(figsize=(10, 3))
graph = nx.grid_2d_graph(1, 3)  # 4x4 grid

# plt.figure(figsize=(w, h))
plt.subplot(131)
g = nx.from_numpy_array(allSC[i])
nx.draw(g, pos, node_color='g', node_size=100, font_size=10, edge_color='k', width=0.25, with_labels=True)
plt.title('Source graph')
# plt.show()

# plt.figure(figsize=(w, h))
plt.subplot(132)
g = nx.from_numpy_array(tfc)
nx.draw(g, pos, node_color='g', node_size=100, font_size=10, edge_color='k', width=0.25, with_labels=True)
plt.title('Empirical target graph')
# plt.show()

# plt.figure(figsize=(w, h))
plt.subplot(133)
g = nx.from_numpy_array(pfc)
nx.draw(g, pos, node_color='g', node_size=100, font_size=10, edge_color='k', width=0.25, with_labels=True,
        title='Pred')
plt.title('Predicted target graph'.format())
plt.savefig('IoT_case40.pdf')
plt.show()