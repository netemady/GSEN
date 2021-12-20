# Created by qli10 at 7/10/2019
import numpy as np
# import torch
from helper import *

def evaluate(predFC,empiricalFC, normalize=True):
    testSize = predFC.shape[0]
    nodeSize = predFC.shape[1]
    # pearsonCorrelations = np.zeros(testSize)
    # diff = np.zeros((testSize, int(nodeSize * (nodeSize-1) / 2)))
    # for row in range(testSize):
    if normalize:
        predfc = getNormalizedMatrix_np(predFC)
        empiricalfc = getNormalizedMatrix_np(empiricalFC)
    else:
        predfc = predFC
        empiricalfc = empiricalFC
    predict_FC_vec = predfc[np.triu(np.ones(predfc.shape),k = 1)==1]#triu2vec(predfc.cpu(), diag=1)
    empirical_FC_vec = empiricalfc[np.triu(np.ones(empiricalfc.shape),k = 1)==1]
    # empirical_FC_vec = triu2vec(empiricalfc, diag=1).detach().cpu().numpy().reshape(1, predict_FC_vec.shape[0])
    predict_fc = predict_FC_vec.reshape((1, predict_FC_vec.shape[0]))
    # diff[row, :] = empirical_FC_vec - predict_fc
    (pearson, p_val) = pearsonr(empirical_FC_vec.flatten(), predict_fc.flatten())
    # pearsonCorrelations = pearson
    return pearson
# In[Xiaojie's implementation]
#
# paper 2014
# ''Abdelnour F, Voss HU, Raj A.
#     Network diffusion accurately models the relationship between structural and functional brain connectivity networks. Neuroimage.
#      2014 Apr 15;90:335-47.''
#
#   paper 2018
#   eigen model in paper:
#     ''Abdelnour F, Dayan M, Devinsky O, Thesen T, Raj A.
#     Functional brain connectivity is predictable from anatomic network's Laplacian eigen-structure. NeuroImage.
#      2018 May 15;172:728-39.''

# def laplacian(mat):
#     rowsum = np.sum(mat, 0)
#     d_inv_sqrt = np.power(rowsum, -0.5)
#     lamda = np.diag(d_inv_sqrt)
#     I = np.eye(mat.shape[1])
#     laplacian = I - np.dot(np.dot(lamda, mat), lamda)
#     return laplacian


def predict_fc_diff(sc, beta_t):
    # sc_lap = laplacian(sc)
    # a,b=np.linalg.eig(sc_lap)
    # a[0]=0
    # a[1]=0
    # sc_lap=np.dot(np.dot(a,b),np.transpose(b))
    fc = np.exp(-1 * beta_t * sc)
    return fc


def predict_fc_eigen(sc, a, beta_t, b):
    # sc_lap = laplacian(sc)
    # a,b=np.linalg.eig(sc_lap)
    # a[0]=0
    # a[1]=0
    # sc_lap=np.dot(np.dot(np.diag(a),b),np.transpose(b))
    fc = a * np.exp(-beta_t * sc) + b * np.eye(sc.shape[1])
    return fc


# fitting the paramters
def curve_fiting_eigen(sc_train, fc_train):
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c

    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(func, sc_train, fc_train,maxfev=10000)
    return popt


def curve_fiting_diff(sc_train, fc_train):
    def func(x, b):
        return np.exp(-b * x)

    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(func, sc_train, fc_train)
    return popt

def preprocessNumpy(rawSC, rawFC, useLaplacian=True, normalized=True, useAbs=False):
    rows = rawSC.shape[0]
    sizeNode = rawSC.shape[1]
    SC = np.zeros((rows, sizeNode, sizeNode))
    SC_u = np.zeros((rows, sizeNode, sizeNode))
    FC_u = np.zeros((rows, sizeNode, sizeNode))
    SC_lamb = np.zeros((rows, sizeNode))
    FC = np.zeros((rows, sizeNode, sizeNode))
    FC_lamb = np.zeros((rows, sizeNode))

    for row in range(rows):
        if useLaplacian:
            sc = getLaplacian_np(rawSC[row], normalized=normalized)
        else:
            sc = rawSC[row]
        lamb_sc, u_sc = np.linalg.eigh(sc)
        SC[row] = sc
        SC_u[row] = u_sc
        SC_lamb[row, :] = lamb_sc
        if useAbs:
            fc = np.abs(rawFC[row])
        else:
            fc = rawFC[row]
        if useLaplacian:
            fc = getLaplacian_np(fc, normalized=normalized)
        FC[row] = fc
        lamb_fc, u_fc = np.linalg.eigh(fc)
        FC_lamb[row, :] = lamb_fc
        FC_u[row] = u_fc
    SC_lamb = SC_lamb.reshape((SC_lamb.shape[0], SC_lamb.shape[1], 1))  # 700*68*1
    return SC,SC_lamb,SC_u,FC,FC_lamb,FC_u
# In[evaluate paper 2014 paper 2018]
def paper20142018(rawData,useLaplacian = True,normalized = False,useAbs = False,trainSize = 700):
    raw_sc_train = rawData['rawSC'][:trainSize,:]
    raw_fc_train = rawData['rawFC'][:trainSize,:]
    trainSC, trainSC_lamb, trainSC_u, trainFC, trainFC_lamb, trainFC_u = preprocessNumpy(raw_sc_train, raw_fc_train,
                                                                                    useLaplacian=useLaplacian,
                                                                                    normalized=normalized,
                                                                                    useAbs=useAbs)
    popt_eigen = curve_fiting_eigen(np.reshape(trainSC_lamb, [-1])[2:], np.reshape(trainFC_lamb, [-1])[2:])
    popt_diff = curve_fiting_diff(np.reshape(trainSC_lamb, [-1])[2:], np.reshape(trainFC_lamb, [-1])[2:])
    #predict fc

    raw_sc_test = rawData['rawSC'][trainSize:, :]
    raw_fc_test = rawData['rawFC'][trainSize:, :]
    testSC, testSC_lamb, testSC_u, empiricalFC, empricalFC_lamb, empiricalFC_u = preprocessNumpy(raw_sc_test, raw_fc_test,
                                                                                             useLaplacian=useLaplacian,
                                                                                             normalized=normalized,
                                                                                             useAbs=useAbs)
    pred_fc_diff=np.zeros((empiricalFC.shape[0],empiricalFC.shape[1],empiricalFC.shape[2]))  # paper 2014
    pred_fc_eigen=np.zeros((empiricalFC.shape[0],empiricalFC.shape[1],empiricalFC.shape[2])) # paper 2018
    corr_diff=np.zeros(empiricalFC.shape[0])
    corr_eigen=np.zeros(empiricalFC.shape[0])
    for i in range(len(testSC)):
        pred_fc_diff[i]=predict_fc_diff(testSC[i],popt_diff[0])
        corr_diff[i] = evaluate(pred_fc_diff[i],empiricalFC[i])
        pred_fc_eigen[i]=predict_fc_eigen(testSC[i],popt_eigen[0],popt_eigen[1],popt_eigen[2])
        corr_eigen[i] = evaluate(pred_fc_eigen[i],empiricalFC[i])

    print ("paper 2014 correlation: %.4f ± %.4f" % (corr_diff.mean(),corr_diff.std()))
    print ("paper 2018 correlation: %.4f ± %.4f" % (corr_eigen.mean(),corr_eigen.std()))
    return corr_diff,corr_eigen


# In[paper 2016]
"""
Created on Wed Apr 17 16:50:25 2019
@author: gxjco

This model is on paper:
    "Meier J, Tewarie P, Hillebrand A, Douw L, van Dijk BW, Stufflebeam SM, Van Mieghem P.
    A mapping between structural and functional brain networks. Brain connectivity. 2016 May 1;6(4):298-311."
"""


def predict_fc_mapping(sc,k,c):
    W=np.zeros((sc.shape[0],sc.shape[1]))
    for i in range(1,k+1):
        W=W+c[i]*np.power(sc,i)
    return W+np.eye(sc.shape[0])*c[0]

def paramter(sc,fc,k):
    W=np.reshape(fc,[-1])
    A=np.zeros((k+1,W.shape[0]))
    for i in range(k+1):
        if i==0:
            A[i]=np.ones(W.shape[0])
        else:
            A[i]=np.reshape(np.power(sc,i),[-1])
    c=np.linalg.lstsq(np.transpose(A),W)[0]
    return c
def paper2016(rawData,useLaplacian = True,normalized = False,useAbs = False,trainSize = 700):

    raw_sc_train = rawData['rawSC'][:trainSize, :]
    raw_fc_train = rawData['rawFC'][:trainSize, :]
    sc_train, trainSC_lamb, trainSC_u, fc_train, trainFC_lamb, trainFC_u = preprocessNumpy(raw_sc_train, raw_fc_train,
                                                                                         useLaplacian=useLaplacian,
                                                                                         normalized=normalized,
                                                                                         useAbs=useAbs)
    raw_sc_test = rawData['rawSC'][trainSize:, :]
    raw_fc_test = rawData['rawFC'][trainSize:, :]
    sc_test, testSC_lamb, testSC_u, fc_test, empricalFC_lamb, empiricalFC_u = preprocessNumpy(raw_sc_test,
                                                                                                 raw_fc_test,
                                                                                                 useLaplacian=useLaplacian,
                                                                                                 normalized=normalized,
                                                                                                 useAbs=useAbs)
    k=3
    c=paramter(sc_train,fc_train,k)
    pred_fc_mapping=np.zeros((sc_test.shape[0],sc_test.shape[1],sc_test.shape[2]))
    corr_mapping=np.zeros(fc_test.shape[0])
    for i in range(sc_test.shape[0]):
       pred_fc_mapping[i]=predict_fc_mapping(sc_test[i],k,c)
       corr_mapping[i] = evaluate(pred_fc_mapping[i], fc_test[i])
    print ("paper 2016 correlation: %.4f ± %.4f" % (corr_mapping.mean(),corr_mapping.std()))
    return corr_mapping
# In[paper 2008]
"""
Created on Wed Apr 17 15:28:53 2019
@author: gxjco

This model is on paper:
    ''Galán RF. On how network architecture determines the dominant patterns of spontaneous neural activity. PloS one.
    2008 May 14;3(5):e2148."

"""


def predict_fc_linear(sc,b,alpha,t):
    A=(1-alpha*t)*np.eye(sc.shape[0])+t*sc
    D,L=np.linalg.eig(A)
    L_inv=np.linalg.inv(L)
    Q=np.eye(sc.shape[0])*b
    Q_star=np.dot(np.dot(L_inv,Q),np.transpose(L_inv))
    P=np.zeros((sc.shape[0],sc.shape[1]))
    for i in range(len(D)):
        for j in range(len(D)):
            P[i][j]=Q_star[i][j]/(1-D[i]*D[j])
    C=np.dot(np.dot(L,P),np.transpose(L))
    return A
def paper2008(rawData,useLaplacian = True,normalized = False,useAbs = False,trainSize = 700):

    raw_sc_train = rawData['rawSC'][:trainSize, :]
    raw_fc_train = rawData['rawFC'][:trainSize, :]
    sc_train, trainSC_lamb, trainSC_u, fc_train, trainFC_lamb, trainFC_u = preprocessNumpy(raw_sc_train, raw_fc_train,
                                                                                         useLaplacian=useLaplacian,
                                                                                         normalized=normalized,
                                                                                         useAbs=useAbs)
    raw_sc_test = rawData['rawSC'][trainSize:, :]
    raw_fc_test = rawData['rawFC'][trainSize:, :]
    sc_test, testSC_lamb, testSC_u, fc_test, empricalFC_lamb, empiricalFC_u = preprocessNumpy(raw_sc_test,
                                                                                                 raw_fc_test,
                                                                                                 useLaplacian=useLaplacian,
                                                                                                 normalized=normalized,
                                                                                                 useAbs=useAbs)


    pred_fc_linear=np.zeros((sc_test.shape[0],sc_test.shape[1],sc_test.shape[2]))
    corr_linear=np.zeros(fc_test.shape[0])
    b = 1
    for i in range(sc_test.shape[0]):
       pred_fc_linear[i] = predict_fc_linear(sc_test[i], b, 2, 100)
       corr_linear[i] = evaluate(pred_fc_linear[i], fc_test[i])
    print ("paper 2008 correlation: %.4f ± %.4f" % (corr_linear.mean(),corr_linear.std()))
    return corr_linear
#
# # In[t-statistics]
# from scipy.stats import ttest_rel
# from scipy import io
# baselines = io.loadmat("results0708.mat")
# ourResults = baselines['results']
# print ("baselines' correlations:\num_of_nodes"
#        "Relu %.4f ± %.4f, \num_of_nodes"
#        "Sigmoid %.4f ± %.4f, \num_of_nodes"
#        "Tanh %.4f ± %.4f, \num_of_nodes"
#        "Softplus %.4f ± %.4f, \num_of_nodes"
#        % (ourResults[:,16].mean(),ourResults[:,16].std(),
#           ourResults[:,17].mean(),ourResults[:,17].std(),
#           ourResults[:,18].mean(),ourResults[:,18].std(),
#           ourResults[:,19].mean(),ourResults[:,19].std()))
# for ourIdx in range(16,20):
#     (t_statistic, p_value) = ttest_rel(ourResults[:,ourIdx],corr_linear)
#     print("%d,t-test paper 2008: t-statistic=%.2e,p=%.2e"
#           %(ourIdx,t_statistic,p_value))
#     (t_statistic, p_value) = ttest_rel(ourResults[:, ourIdx], corr_mapping)
#     print("%d,t-test paper 2008: t-statistic=%.2e,p=%.2e"
#           % (ourIdx, t_statistic, p_value))
#     (t_statistic, p_value) = ttest_rel(ourResults[:, ourIdx], corr_diff)
#     print("%d,t-test paper 2008: t-statistic=%.2e,p=%.2e"
#           % (ourIdx, t_statistic, p_value))
#     (t_statistic, p_value) = ttest_rel(ourResults[:, ourIdx], corr_eigen)
#     print("%d,t-test paper 2008: t-statistic=%.2e,p=%.2e"
#           % (ourIdx, t_statistic, p_value))
# t_stats = np.zeros((4,8))
# for ourIdx in range(16,20):
#     (t_statistic, p_value) = ttest_rel(ourResults[:,ourIdx],corr_linear)
#     print("%d,t-test paper 2008: %.2e %.2e"
#           %(ourIdx,t_statistic,p_value))
#     t_stats[ourIdx-16,0:2] = np.array([t_statistic,p_value])
#
#     (t_statistic, p_value) = ttest_rel(ourResults[:, ourIdx], corr_mapping)
#     print("%d,t-test paper 2008: %.2e %.2e"
#           % (ourIdx, t_statistic, p_value))
#     t_stats[ourIdx-16,2:4] = np.array([t_statistic,p_value])
#
#     (t_statistic, p_value) = ttest_rel(ourResults[:, ourIdx], corr_diff)
#     print("%d,t-test paper 2008: %.2e %.2e"
#           % (ourIdx, t_statistic, p_value))
#     t_stats[ourIdx-16,4:6] = np.array([t_statistic,p_value])
#
#     (t_statistic, p_value) = ttest_rel(ourResults[:, ourIdx], corr_eigen)
#     print("%d,t-test paper 2008: %.2e %.2e"
#           % (ourIdx, t_statistic, p_value))
#     t_stats[ourIdx-16,6:8] = np.array([t_statistic,p_value])
# print(t_stats)

# In[]

trainSize = 700
rawData = np.load('original_sc_fc_corr.npz')
paper20142018(rawData,useLaplacian = True,normalized = True,useAbs = False,trainSize = 700)
# paper2016(rawData,useLaplacian = True,normalized = True,useAbs = False,trainSize = 700)
# paper2008(rawData,useLaplacian = True,normalized = True,useAbs = False,trainSize = 700)
