# Created by qli10 at 8/16/2019
# purely 2d convolution

import torch
import torch.nn as nn
from a0813run_model import preprocess,evaluate,loss_correlation
import numpy as np


# In[a_net0816_corr_tune]
class spectralNet(nn.Module):
# class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

    def __init__(self,K=5,num_hidden_layer=10,activation=1, debug=False):
        super(spectralNet, self).__init__()
        self.K = K
        self.in_channel_hidden = K
        self.out_channel_hidden = K
        self.num_hidden_layer = num_hidden_layer
        self.debug = debug
        self.activation = activation
        self.conv_sigmoid = nn.Sequential(  # input shape (1, 1, 68,K)
            nn.Conv2d(
                in_channels=1,  #
                out_channels=1,  # 0
                kernel_size=(1, self.K),  # filter size
                stride=1,  # filter movement/step
                padding=0,  # no need padding
                bias=True,
            ),  # output shape (1,1,68,1)
            nn.Sigmoid(),  # activation
            # nn.ReLU(),  # activation
            # nn.Tanh(),
            # nn.Softplus(),
        )
        self.conv_relu = nn.Sequential(  # input shape (1, 1, 68,K)
            nn.Conv2d(
                in_channels=1,  #
                out_channels=1,  #0
                kernel_size= (1,self.K),  # filter size
                stride=1,  # filter movement/step
                padding=0,  # no need padding
                bias=True,
            ),  # output shape (1,1,68,1)
            # nn.Sigmoid(),  # activation
            nn.ReLU(),  # activation
            # nn.Tanh(),
            # nn.Softplus(),
        )
        self.conv_tanh = nn.Sequential(  # input shape (1, 1, 68,K)
            nn.Conv2d(
                in_channels=1,  #
                out_channels=1,  # 0
                kernel_size=(1, self.K),  # filter size
                stride=1,  # filter movement/step
                padding=0,  # no need padding
                bias=True,
            ),  # output shape (1,1,68,1)
            # nn.Sigmoid(),  # activation
            # nn.ReLU(),  # activation
            nn.Tanh(),
            # nn.Softplus(),
        )
        self.conv_softplus = nn.Sequential(  # input shape (1, 1, 68,K)
            nn.Conv2d(
                in_channels=1,  #
                out_channels=1,  # 0
                kernel_size=(1, self.K),  # filter size
                stride=1,  # filter movement/step
                padding=0,  # no need padding
                bias=True,
            ),  # output shape (1,1,68,1)
            # nn.Sigmoid(),  # activation
            # nn.ReLU(),  # activation
            # nn.Tanh(),
            nn.Softplus(),
        )

        # self.conv_out = nn.Sequential(  # input shape (1, 1, 68)
        #     nn.Conv2d(
        #         in_channels=1,  #
        #         # out_channels=self.out_channel_hidden,  #
        #         out_channels=1,  # 0
        #         kernel_size=1,  # filter size
        #         stride=1,  # filter movement/step
        #         padding=0,  # no need padding
        #         # groups=self.in_channel_hidden,
        #         bias=True,
        #     ),  # output shape (16,68,1)
        # )

        # self.conv_out1 = nn.Sequential(  # input shape (1, 1, 68)
        #     nn.Linear(
        #         in_channels=1,  #
        #         # out_channels=self.out_channel_hidden,  #
        #         out_channels=1,  # 0
        #         kernel_size=1,  # filter size
        #         stride=1,  # filter movement/step
        #         padding=0,  # no need padding
        #         # groups=self.in_channel_hidden,
        #         bias=True,
        #     ),  # output shape (16,68,1)
        #     # nn.Sigmoid(),  # activation
        #     # nn.ReLU(),  # activation
        #     # nn.Tanh(),
        #     # nn.Softplus(),
        # )


    def forward(self, x,u):
        # self.u = u
        # x = x.transpose(1,2)  #from 700*68*1 to 700*1*68
        # x = torch.cat([x**i for i in range(1,self.K+1)],3)  # x.shape = 700 * 1 * 68 * 5
        # if self.debug:
        #     print('input_layer1.shape:',x.shape)
        #     print('x_input:',x[0,0,:])
        #     # print('x_input:', x[0, 1, :])
        # x = self.conv_hidden(x)
        # x = torch.cat([x**i for i in range(1,self.K+1)],3)   # x.shape = 700 * 1 * 68 * 5
        for layer in range(self.num_hidden_layer):
            x = torch.cat([x ** i for i in range(1, self.K + 1)], 3)
            if self.activation==1:
                x = self.conv_sigmoid(x)
            elif self.activation==2:
                x = self.conv_relu(x)
            else:
                x = self.conv_tanh(x)
            # x = x + 1


        # x = 2*x


        # x = torch.cat([x ** i for i in range(1, self.K + 1)], 3)
        # x = self.conv_out(x)
        # x = x.view(x.shape[0],1,1,68)
        # x = x.transpose(2,3)
        # number_of_edgs = nn.Linear(68,68)
        # number_of_edgs.cuda()
        # x = number_of_edgs(x)
        # x = x.transpose(2,3)
        # print ('out.shape',x.shape)
        # x = x.view(x.shape[0],1,x.shape[1],x.shape[1])
        # # use conv
        # x = x.transpose(2, 1)
        # x = self.conv_out(x)
        x = x.expand(x.shape[0], x.shape[1],  x.shape[2], x.shape[2])  # copy lambda vector to diag(lambda)
        if torch.cuda.is_available():
            x = torch.mul(x,torch.eye(x.shape[2]).cuda())
        else:
            x = torch.mul(x,torch.eye(x.shape[2]))

        # output = torch.matmul(torch.matmul(self.u,x.view(x.shape[0],x.shape[2],x.shape[3])),self.u.transpose(1,2))
        output = torch.matmul(torch.matmul(u,x.view(x.shape[0],x.shape[2],x.shape[3])),u.transpose(1,2))
        # x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 68)
        return output, x  # return x for visualization


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
# In[]
trainSize = 700
# datasetName = 'original_sc_fc_partial.npz'
datasetName = 'original_sc_fc_corr.npz'

# rawData = np.load('original_sc_fc_corr.npz')
rawData = np.load(datasetName)

useLaplacian, normalized, useAbs = True, True, False
rawSC = rawData['rawSC']
rawFC = rawData['rawFC']
trainSCraw = torch.from_numpy(rawSC[:trainSize, :]).float()
trainFCraw = torch.from_numpy(rawFC[:trainSize, :]).float()
trainSC, trainSC_lamb, trainSC_u, trainFC, trainFC_lamb, trainFC_u = preprocess(trainSCraw, trainFCraw,
                                                                                useLaplacian=useLaplacian,
                                                                                normalized=normalized,
                                                                                useAbs=useAbs)
trainSC_lamb = trainSC_lamb.view(trainSC_lamb.shape[0],1,trainSC_lamb.shape[1],trainSC_lamb.shape[2])


testSCraw = torch.from_numpy(rawSC[trainSize:, :]).float()
testFCraw = torch.from_numpy(rawFC[trainSize:, :]).float()

testSC, testSC_lamb, testSC_u, empiricalFC, empricalFC_lamb, empiricalFC_u = preprocess(testSCraw, testFCraw,
                                                                                        useLaplacian=useLaplacian,
                                                                                        normalized=normalized,
                                                                                        useAbs=useAbs)
testSC_lamb = testSC_lamb.view(testSC_lamb.shape[0],1,testSC_lamb.shape[1],testSC_lamb.shape[2])


activations = range(1,4)
layers = range(1,11)
Ks = range(1,11)
import itertools
combinations = list(itertools.product(*[activations,layers,Ks]))
result = np.zeros((len(combinations),6))  # column names: 0: i, 1:activation, 2: layer, 3: K, 4: mean, 5: std
# from a0813Net_func import spectralNet
for i in range(len(combinations)):
    activation,layer,K = combinations[i]

    net = spectralNet(K=K,num_hidden_layer=layer,activation=activation)
    if torch.cuda.is_available():
        print('Using GPU~~~~~~~~~~~~~~~')
        net.cuda()
    else:
        print('Using CPU  ~~~~~~~~~~~~~~~')
    ## In[] #########  Train the network ################
    net, loss = trainNet(net, trainSC_lamb, trainSC_u, trainFC,
                         maxIters=5000, lr=1e-2, useCorrLoss=True, FC_u=None)
    #########  Test the results  ################

    predFC, FC_vec = net(testSC_lamb, testSC_u)
    # predFC,FC_vec = net(testSC_lamb, empiricalFC_u)   #debug only

    #########  Evaluate the results ##############
    pearsonCorrelation, diff = evaluate(predFC, empiricalFC)
    # try:
    #     ## In[] #########  Train the network ################
    #     net, loss = trainNet(net, trainSC_lamb, trainSC_u, trainFC,
    #                          maxIters=50, lr=1e-2, useCorrLoss=True, FC_u=None)
    #     #########  Test the results  ################
    #
    #     predFC, FC_vec = net(testSC_lamb, testSC_u)
    #     # predFC,FC_vec = net(testSC_lamb, empiricalFC_u)   #debug only
    #
    #     #########  Evaluate the results ##############
    #     pearsonCorrelation, diff = evaluate(predFC, empiricalFC)
    # except Exception as err:
    #     print("######## Our Throw Excepation: ",err,combinations[i])
    #     pearsonCorrelation = np.zeros(1)
    print('mean pearson:', pearsonCorrelation.mean(), pearsonCorrelation.std())
    result[i,:] = np.asanyarray([i,activation,layer,K,pearsonCorrelation.mean(),pearsonCorrelation.std()])

print('result',result)
np.savez('result_'+datasetName,result=result)