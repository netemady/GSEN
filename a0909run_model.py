import numpy as np
import torch
from helper import *

# source.shape = (batchSize,nodeSize,nodeSize)
# source.shape = (batchSize,nodeSize,nodeSize)
# return loss based on the correlation between the elements of upper triangle elements
def loss_correlation(source, target):
    row_idx, col_idx = torch.triu_indices(source.shape[1], source.shape[2],offset=1)
    x = source[:, row_idx, col_idx]
    y = target[:, row_idx, col_idx]
    vx = x - torch.mean(x, 1, keepdim=True)
    vy = y - torch.mean(y, 1, keepdim=True)
    xy_cov = torch.bmm(vx.view(vx.shape[0], 1, vx.shape[1]), vy.view(vy.shape[0], vy.shape[1], 1), ).view(vx.shape[0])
    cost = xy_cov / torch.mul(torch.sqrt(torch.sum(vx ** 2, dim=1)), torch.sqrt(torch.sum(vy ** 2, dim=1)))
    loss = 1 - torch.mean(cost)
    return loss

###  load brain SC_FCcorr_FCpartcorr dataset
def trainTestDataBrain(allData,trainSize = 700,sameDataset = False):
    testSize = allData.shape[0] - trainSize
    trainData = torch.from_numpy(allData[:trainSize, :]).float()
    if not sameDataset:
        testData = torch.from_numpy(allData[trainSize:, :]).float()
    else:
        testSize=trainSize;testData = torch.from_numpy(allData[:testSize ,:]).float()   # for debug only, overfitting
    return trainData,testData

def preprocess(rawSC, rawFC, useLaplacian=True, normalized=True, useAbs=False):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    rows = rawSC.shape[0]
    sizeNode = rawSC.shape[1]
    SC = torch.zeros(rows, sizeNode, sizeNode, dtype=torch.float32, device=device)
    SC_u = torch.zeros(rows, sizeNode, sizeNode, dtype=torch.float32, device=device)
    FC_u = torch.zeros(rows, sizeNode, sizeNode, dtype=torch.float32, device=device)
    SC_lamb = torch.zeros(rows, sizeNode, dtype=torch.float32, device=device)
    FC = torch.zeros(rows, sizeNode, sizeNode, dtype=torch.float32, device=device)
    FC_lamb = torch.zeros(rows, sizeNode, dtype=torch.float32, device=device)

    for row in range(rows):
        if useLaplacian:
            sc = getLaplacian(rawSC[row], normalized=normalized)
        else:
            sc = rawSC[row]
        lamb_sc, u_sc = torch.symeig(sc, eigenvectors=True)
        SC[row] = sc
        SC_u[row] = u_sc
        SC_lamb[row, :] = lamb_sc
        if useAbs:
            fc = torch.abs(rawFC[row])
        else:
            fc = rawFC[row]
        if useLaplacian:
            fc = getLaplacian(fc, normalized=normalized)
        FC[row] = fc
        lamb_fc, u_fc = torch.symeig(fc, eigenvectors=True)
        FC_lamb[row, :] = lamb_fc
        FC_u[row] = u_fc
    SC_lamb = SC_lamb.reshape((SC_lamb.shape[0], SC_lamb.shape[1], 1))  # 700*68*1
    return SC,SC_lamb,SC_u,FC,FC_lamb,FC_u


def preprocessData(data, useLaplacian=True, normalized=True, useAbs=False):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    rows = data.shape[0]
    sizeConnectivity = int(data.shape[1]/2)

    sizeNode = int(np.sqrt(sizeConnectivity))
    SC = torch.zeros(rows, sizeNode, sizeNode, dtype=torch.float32, device=device)
    SC_u = torch.zeros(rows, sizeNode, sizeNode, dtype=torch.float32, device=device)
    FC_u = torch.zeros(rows, sizeNode, sizeNode, dtype=torch.float32, device=device)
    SC_lamb = torch.zeros(rows, sizeNode, dtype=torch.float32, device=device)
    FC = torch.zeros(rows, sizeNode, sizeNode, dtype=torch.float32, device=device)
    FC_lamb = torch.zeros(rows, sizeNode, dtype=torch.float32, device=device)

    for row in range(rows):
        sc_A = data[row, :sizeConnectivity].reshape((sizeNode, sizeNode))
        if useLaplacian:
            SC[row] = getLaplacian(sc_A, normalized=normalized)
        else:
            SC[row] = sc_A
        lamb_sc, u_sc = torch.symeig(SC[row], eigenvectors=True)
        SC_u[row] = u_sc
        SC_lamb[row, :] = lamb_sc
        if useAbs:
            fc_A = torch.abs(data[row, sizeConnectivity: 2 * sizeConnectivity].reshape((sizeNode, sizeNode)))
        else:
            fc_A = (data[row, sizeConnectivity: 2 * sizeConnectivity].reshape((sizeNode, sizeNode)))
        if useLaplacian:
            fc = getLaplacian(fc_A, normalized=normalized)
        else:
            fc = fc_A
        FC[row] = fc
        lamb_fc, u_fc = torch.symeig(fc, eigenvectors=True)
        FC_lamb[row, :] = lamb_fc
        FC_u[row] = u_fc
    SC_lamb = SC_lamb.reshape((SC_lamb.shape[0], SC_lamb.shape[1], 1))  # 700*68*1
    return SC,SC_lamb,SC_u,FC,FC_lamb,FC_u


def trainNet(net, SC_lamb,SC_u,FC,maxIters=200, lr = 1e-2,useCorrLoss=True,FC_u=None):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    print('Neual Network Structure:\n',net)
    for t in range(maxIters):
        if FC_u==None:
            train_predict_FC, fc_vec = net(SC_lamb, SC_u)
        else:
            train_predict_FC, fc_vec = net(SC_lamb, FC_u)  ############### debug only
        if useCorrLoss:
            loss = loss_correlation(train_predict_FC, FC)
        else:
            loss_func = torch.nn.MSELoss()
            loss = loss_func(train_predict_FC, FC)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if t % 50 == 0:
            print("Loss = ",loss, t)
    return net,loss


def evaluate(predFC,empiricalFC, normalize=True):
    testSize = predFC.shape[0]
    nodeSize = predFC.shape[1]
    pearsonCorrelations = np.zeros(testSize)
    diff = np.zeros((testSize, int(nodeSize * (nodeSize-1) / 2)))
    for row in range(testSize):
        if normalize:
            predfc = getNormalizedMatrix(predFC[row])
            empiricalfc = getNormalizedMatrix(empiricalFC[row])
        else:
            predfc = predFC[row]
            empiricalfc = empiricalFC[row]
        predict_FC_vec = triu2vec(predfc.cpu(), diag=1)
        empirical_FC_vec = triu2vec(empiricalfc, diag=1).detach().cpu().numpy().reshape(1, predict_FC_vec.shape[0])
        predict_fc = predict_FC_vec.detach().numpy().reshape(1, predict_FC_vec.shape[0])
        diff[row, :] = empirical_FC_vec - predict_fc
        (pearson, p_val) = pearsonr(empirical_FC_vec.flatten(), predict_fc.flatten())
        pearsonCorrelations[row] = pearson
    return pearsonCorrelations,diff



def run_our_model0909(rawData,useLaplacian = True,normalized = False,useAbs = False,trainSize = 700):
    rawSC = rawData['rawSC']
    rawFC = rawData['rawFC']
    trainSCraw = torch.from_numpy(rawSC[:trainSize, :]).float()
    trainFCraw = torch.from_numpy(rawFC[:trainSize, :]).float()
    trainSC, trainSC_lamb, trainSC_u, trainFC, trainFC_lamb, trainFC_u = preprocess(trainSCraw, trainFCraw,
                                                                                    useLaplacian=useLaplacian,
                                                                                    normalized=normalized,
                                                                                    useAbs=useAbs)

    from a0813Net_func import spectralNet
    net = spectralNet()
    if torch.cuda.is_available():
        print('Using GPU~~~~~~~~~~~~~~~')
        net.cuda()
    else:
        print('Using CPU  ~~~~~~~~~~~~~~~')
    #########  Train the network ################
    net, loss = trainNet(net, trainSC_lamb, trainSC_u, trainFC,
                         maxIters=2000, lr=1e-3, useCorrLoss=True, FC_u=None)
    #########  Test the results  ################
    testSCraw = torch.from_numpy(rawSC[trainSize:, :]).float()
    testFCraw = torch.from_numpy(rawFC[trainSize:, :]).float()
    testSC, testSC_lamb, testSC_u, empiricalFC, empricalFC_lamb, empiricalFC_u = preprocess(testSCraw, testFCraw,
                                                                                            useLaplacian=useLaplacian,
                                                                                            normalized=normalized,
                                                                                            useAbs=useAbs)

    predFC, FC_vec = net(testSC_lamb, testSC_u)
    # predFC,FC_vec = net(testSC_lamb, empiricalFC_u)   #debug only

    #########  Evaluate the results ##############
    pearsonCorrelation, diff = evaluate(predFC, empiricalFC)
    print('mean pearson:', pearsonCorrelation.mean(), pearsonCorrelation.std())
    return pearsonCorrelation,diff

