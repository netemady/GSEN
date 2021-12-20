# Created by qli10 at 9/9/2019
# In[]
import numpy as np
import torch
from sklearn.model_selection import KFold
import itertools
from helper import *
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
                x = 2*x
            elif self.activation==2:
                x = self.conv_relu(x)
            else:
                x = self.conv_tanh(x)
                x = x + 1


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



# In[]
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
def loss_correlation(source, target):
    row_idx, col_idx = torch.triu_indices(source.shape[1], source.shape[2],offset=1)
    x = source[:, row_idx, col_idx]
    y = target[:, row_idx, col_idx]
    vx = x - torch.mean(x, 1, keepdim=True)
    vy = y - torch.mean(y, 1, keepdim=True)
    xy_cov = torch.bmm(vx.view(vx.shape[0], 1, vx.shape[1]), vy.view(vy.shape[0], vy.shape[1], 1), ).view(vx.shape[0])
    cost = xy_cov / torch.mul(torch.sqrt(torch.sum(vx ** 2, dim=1)), torch.sqrt(torch.sum(vy ** 2, dim=1)))
    loss = 1 - torch.mean(cost)
    # print('loss:',loss)
    return loss
def trainNet(net, SC_lamb,SC_u,FC,maxIters=200, lr = 1e-2,useCorrLoss=True,FC_u=None):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # print('Neual Network Structure:\num_of_nodes',net)
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
        # if t % 50 == 0:
        #     print("Loss = ",loss, t)
    return net,loss
# In[]
# both SC and FC should be in laplacian format
def preprocess(SC, FC):
    rows = SC.shape[0]
    sizeNode = SC.shape[1]
    device=SC.device
    SC_u = torch.zeros(rows, sizeNode, sizeNode, dtype=torch.float32, device=device)
    FC_u = torch.zeros(rows, sizeNode, sizeNode, dtype=torch.float32, device=device)
    SC_lamb = torch.zeros(rows, sizeNode, dtype=torch.float32, device=device)
    # FC = torch.zeros(rows, sizeNode, sizeNode, dtype=torch.float32, device=device)
    FC_lamb = torch.zeros(rows, sizeNode, dtype=torch.float32, device=device)

    for row in range(rows):
        # if useLaplacian:
        #     sc = getLaplacian(rawSC[row], normalized=normalized)
        # else:
        #     sc = rawSC[row]
        sc = SC[row]
        lamb_sc, u_sc = torch.symeig(sc, eigenvectors=True)
        # SC[row] = sc
        SC_u[row] = u_sc
        SC_lamb[row, :] = lamb_sc
        # if useAbs:
        #     fc = torch.abs(rawFC[row])
        # else:
        #     fc = rawFC[row]
        # if useLaplacian:
        #     fc = getLaplacian(fc, normalized=normalized)
        # FC[row] = fc
        fc = FC[row]
        lamb_fc, u_fc = torch.symeig(fc, eigenvectors=True)
        FC_lamb[row, :] = lamb_fc
        FC_u[row] = u_fc
    SC_lamb = SC_lamb.reshape((SC_lamb.shape[0], SC_lamb.shape[1], 1))  # 700*68*1
    return SC,SC_lamb,SC_u,FC,FC_lamb,FC_u

def train_best_spectralNet(sc_train,sc_test,
                         fc_train,fc_test,
                         type='N'):
    activations = range(0, 2)
    layers = range(1, 11,3)
    Ks = range(1, 11,3)

    # activations = range(1,4)
    # layers = range(1,8,2)
    # Ks = range(1,11,4)

    # activations = range(0,1)
    # layers = range(1,2)
    # Ks = range(2,3)


    trainSC, trainSC_lamb, trainSC_u, trainFC, trainFC_lamb, trainFC_u = preprocess(sc_train, fc_train)
    trainSC_lamb = trainSC_lamb.view(trainSC_lamb.shape[0], 1, trainSC_lamb.shape[1], trainSC_lamb.shape[2])
    testSC, testSC_lamb, testSC_u, testFC, testFC_lamb, testFC_u = preprocess(sc_test, fc_test)
    testSC_lamb = testSC_lamb.view(testSC_lamb.shape[0],1,testSC_lamb.shape[1],testSC_lamb.shape[2])


    combinations = list(itertools.product(*[activations, layers, Ks]))
    best_pearsonCorrelation = 0
    for i in range(len(combinations)):
        activation,layer,K = combinations[i]

        net = spectralNet(K=K,num_hidden_layer=layer,activation=activation)
        best_net = net
        if torch.cuda.is_available():
            # print('Using GPU~~~~~~~~~~~~~~~')
            net.cuda()
        else:
            print('Using CPU  ~~~~~~~~~~~~~~~')
        ## In[] #########  Train the network ################
        net, loss = trainNet(net, trainSC_lamb, trainSC_u, trainFC,
                             maxIters=200, lr=1e-2, useCorrLoss=True, FC_u=None)
        #########  Test the results  ################

        predFC, FC_vec = net(testSC_lamb, testSC_u)
        # predFC,FC_vec = net(testSC_lamb, empiricalFC_u)   #debug only

        #########  Evaluate the results ##############
        pearsonCorrelation, diff = evaluate(predFC, fc_test)
        meanPearson = np.mean(pearsonCorrelation)
        if(meanPearson>best_pearsonCorrelation):
           best_pearsonCorrelation = meanPearson
           best_net = net
    return best_net
# In[]
def train_spectralNet(sc_train, fc_train, K=5, layer=3,activation=0):
    trainSC, trainSC_lamb, trainSC_u, trainFC, trainFC_lamb, trainFC_u = preprocess(sc_train, fc_train)
    trainSC_lamb = trainSC_lamb.view(trainSC_lamb.shape[0], 1, trainSC_lamb.shape[1], trainSC_lamb.shape[2])
    net = spectralNet(K=K, num_hidden_layer=layer, activation=activation)
    best_net = net
    if torch.cuda.is_available():
        # print('Using GPU~~~~~~~~~~~~~~~')
        net.cuda()
    else:
        print('Using CPU  ~~~~~~~~~~~~~~~')
    ## In[] #########  Train the network ################
    net, loss = trainNet(net, trainSC_lamb, trainSC_u, trainFC,
                         maxIters=3000, lr=1e-2, useCorrLoss=True, FC_u=None)
    return net

#In[]
# from a_net0816_corr_tune import spectralNet
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
folder = 'synDatasets/'
files = ['syn_kernel_com_L','syn_kernel_com_Z','syn_kernel_exp_N','syn_kernel_gen_L',
         'syn_kernel_gen_Z','syn_kernel_heat_L','syn_kernel_heat_Z','syn_kernel_neu_N',
         'syn_kernel_path_N','syn_kernel_reg_L','syn_kernel_reg_L']
# file = files[4]
for file in files:
    data = np.load(folder+file+'.npz')
    # rawSC = torch.from_numpy(data['sc_A'])
    rawSC = torch.from_numpy(data['sc'])
    rawFC = torch.from_numpy(data['fc'])


    # folder = '../SC_FC_dataset_0905/'
    # file = 'SC_GAMBLING_partialCorrelationFC_L1.npz'
    # data = np.load(folder+file)
    # rawSC = torch.from_numpy(data['sc'])
    # rawFC = torch.from_numpy(data['fc'])

    # folder = '../SC_FC_dataset_0905/'
    # file = 'SC_RESTINGSTATE_correlationFC.npz'
    # data = np.load(folder+file)
    # rawSC = torch.from_numpy(data['rawSC'])
    # rawFC = torch.from_numpy(data['rawFC'])

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        # device = 'cuda:1'


    allSC = torch.zeros(rawSC.shape, dtype=torch.float32, device=device)
    allFC = torch.zeros(rawFC.shape, dtype=torch.float32, device=device)
    for i in range(rawFC.shape[0]):
        allSC[i] = getLaplacian(rawSC[i],normalized=True)
        allFC[i] = getLaplacian(rawFC[i],normalized=True)
    cvFolers = 5
    kf = KFold(n_splits=cvFolers)
    pearsonCorrelation = np.zeros(cvFolers)
    # kf.get_n_splits(allSC)
    # print(kf)
    cv_idx = 0
    for train_index, test_index in kf.split(allSC):
        # print("TRAIN:", train_index, "TEST:", test_index)
        sc_train, sc_test = allSC[train_index], allSC[test_index]
        fc_train, fc_test = allFC[train_index], allFC[test_index]
        kf_validation = KFold(n_splits=cvFolers)
        for validation_train_index,validation_test_index in kf_validation.split(sc_train):
            sc_validation_train,sc_validation_test = sc_train[validation_train_index],sc_train[validation_test_index]
            fc_validation_train,fc_validation_test = fc_train[validation_train_index],fc_train[validation_test_index]
            bestnet = train_best_spectralNet(sc_validation_train,sc_validation_test,
                                         fc_validation_train,fc_validation_test,
                                         type='N')

            break
        net = train_spectralNet(sc_train, fc_train,K=bestnet.K,layer=bestnet.num_hidden_layer,activation=bestnet.activation)

        #####################  Test ########################
        testSC, testSC_lamb, testSC_u, testFC, testFC_lamb, testFC_u = preprocess(sc_test, fc_test)
        testSC_lamb = testSC_lamb.view(testSC_lamb.shape[0], 1, testSC_lamb.shape[1], testSC_lamb.shape[2])
        predFC, FC_vec = net(testSC_lamb,testSC_u)
        correlations, diff = evaluate(predFC, testFC)
        pearsonCorrelation[cv_idx] = np.mean(correlations)
        print(file+' '+'{:.2f}'.format(pearsonCorrelation[cv_idx]))
        cv_idx = cv_idx+1

        # break
    outfile = folder + 'result_sc_' + file
    # print(outfile,np.mean(pearsonCorrelation))
    outtxt = outfile + ' ' + '{:.2f}'.format(np.mean(pearsonCorrelation))
    print(outtxt)
    file1 = open(folder + 'result_sc.txt', "a")
    file1.write(outtxt + '\n')
    file1.close()
    np.save(outfile, pearsonCorrelation)

