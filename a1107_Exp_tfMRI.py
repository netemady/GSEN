# Created by qli10 at 9/10/2019

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


    def forward(self, x,u,D, beta=0):
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
            x = torch.mul(x,torch.pow(D,-beta))
            x = torch.cat([x ** i for i in range(1, self.K + 1)], 3)
            if self.activation==1:
                x = self.conv_sigmoid(x)
                x = 2*x
            elif self.activation==2:
                x = self.conv_relu(x)
            else:
                x = self.conv_tanh(x)
                x = x + 1
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
def tonp(predFC,empiricalFC, normalize=True):
    testSize = predFC.shape[0]
    nodeSize = predFC.shape[1]
    predictions = np.zeros((testSize,int(nodeSize*(nodeSize-1)/2)))
    labels = np.zeros((testSize,int(nodeSize*(nodeSize-1)/2)))

    # pearsonCorrelations = np.zeros(testSize)
    # diff = np.zeros((testSize, int(nodeSize * (nodeSize - 1) / 2)))
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
    # return predict_fc, empirical_FC_vec
        predictions[row] = predict_fc
        labels[row] = empirical_FC_vec
    return predictions, labels

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
    # print('loss_correlation shape:',source.shape,target.shape)
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


def trainNet(net, SC_lamb,SC_u,FC,D,beta=0,maxIters=2001, lr=1e-3,useCorrLoss=True,FC_u=None):

    # loss=10
    # attempt = 0
    # while loss>0.1 and attempt<20:
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.09)
    # optimizer = torch.optim.Adagrad(net.parameters(), lr=lr, lr_decay=0, weight_decay=0)
    # optimizer = torch.optim.Rprop(net.parameters(), lr=lr, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
    # print('Neual Network Structure:\num_of_nodes',net)
    # attempt = attempt+1
    for t in range(maxIters):
        if FC_u==None:
            train_predict_FC, fc_vec = net(SC_lamb, SC_u,D,beta=beta)
        else:
            train_predict_FC, fc_vec = net(SC_lamb, FC_u,D,beta=beta)  ############### debug only
        if useCorrLoss:
        # if False:
            loss = loss_correlation(train_predict_FC, FC)
        else:
            loss_func = torch.nn.MSELoss()
            loss = loss_func(train_predict_FC, FC)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 10 == 0:
            print("Loss = ",loss, t)
        if loss<1e-2 and t>0.2*maxIters:
            print('Good news! Stop early on current training:',loss,t)
            return net, loss

        if t%500==0 and loss>0.75:
            print('Bad news! Failed on current training:',loss,t)
            return net,loss
    return net,loss
# In[]
# both SC and FC should be in laplacian format
def preprocess(SC, FC):
    rows = SC.shape[0]
    sizeNode = SC.shape[1]
    device=SC.device
    SC_u = torch.zeros(rows, sizeNode, sizeNode, dtype=torch.float32, device=device)
    SC_D = torch.zeros(rows, sizeNode, dtype=torch.float32, device=device)
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
        A = SC[row]
        diagIndices = torch.arange(0, A.shape[0])
        A[diagIndices, diagIndices] = 0
        D_vec = torch.sum(torch.abs(A), dim=0)
        SC_D[row] = D_vec

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
    return SC,SC_lamb,SC_u,FC,FC_lamb,FC_u,SC_D

def train_best_spectralNet(sc_train,sc_test,
                         fc_train,fc_test,
                         type='N'):
    # activations = range(0, 2)
    # layers = range(2,4)
    # Ks = range(3, 6,2)

    activations = range(0,1)
    layers = range(3, 4)
    Ks = range(5, 6, 2)



    trainSC, trainSC_lamb, trainSC_u, trainFC, trainFC_lamb, trainFC_u, trainSC_D = preprocess(sc_train, fc_train)
    trainSC_D = trainSC_lamb.view(trainSC_lamb.shape[0], 1, trainSC_lamb.shape[1], trainSC_lamb.shape[2])
    trainSC_lamb = trainSC_lamb.view(trainSC_lamb.shape[0], 1, trainSC_lamb.shape[1], trainSC_lamb.shape[2])
    testSC, testSC_lamb, testSC_u, testFC, testFC_lamb, testFC_u, testSC_D = preprocess(sc_test, fc_test)
    testSC_D = testSC_lamb.view(testSC_lamb.shape[0],1,testSC_lamb.shape[1],testSC_lamb.shape[2])
    testSC_lamb = testSC_lamb.view(testSC_lamb.shape[0],1,testSC_lamb.shape[1],testSC_lamb.shape[2])


    combinations = list(itertools.product(*[activations, layers, Ks]))
    best_pearsonCorrelation = 0
    for i in range(len(combinations)):
        activation,layer,K = combinations[i]
        print('hyperparameters:', K, layer, activation)
        loss=10
        attempt = 0
        minLoss = 100
        while loss>0.7 and attempt<5:
            attempt=attempt+1
            net = spectralNet(K=K,num_hidden_layer=layer,activation=activation)
            # best_net = net

            if torch.cuda.is_available():
                # print('Using GPU~~~~~~~~~~~~~~~')
                net.cuda()
            else:
                print('Using CPU  ~~~~~~~~~~~~~~~')
            ## In[] #########  Train the network ################
            trained_net, loss = trainNet(net, trainSC_lamb, trainSC_u, trainFC,trainSC_D,beta=beta,
                                 maxIters=2001, lr=1e-3, useCorrLoss=True, FC_u=None)
            #########  Test the results  ################

            predFC, FC_vec = trained_net(testSC_lamb, testSC_u,testSC_D,beta=beta)
            # predFC,FC_vec = net(testSC_lamb, empiricalFC_u)   #debug only

            #########  Evaluate the results ##############
            pearsonCorrelation, diff = evaluate(predFC, fc_test)
            meanPearson = np.mean(pearsonCorrelation)
            if(not np.isnan(meanPearson)) and meanPearson>best_pearsonCorrelation:
                best_pearsonCorrelation = meanPearson
                best_net = trained_net
                minLoss = loss
                # print('Current best_net hyperparameters:', best_net.K, best_net.num_hidden_layer, best_net.activation,
                #       best_pearsonCorrelation, minLoss)
            # else:
            #
            #     print('Current hyperparameters:', net.K, net.num_hidden_layer, net.activation,
            #           meanPearson, loss)

    print('Final best_net hyperparameters:',best_net.K,best_net.num_hidden_layer,best_net.activation,best_pearsonCorrelation,minLoss)
    return best_net
# In[]
def train_spectralNet(sc_train, fc_train, K=5, layer=2,activation=0):
    trainSC, trainSC_lamb, trainSC_u, trainFC, trainFC_lamb, trainFC_u, trainSC_D = preprocess(sc_train, fc_train)
    trainSC_D = trainSC_lamb.view(trainSC_lamb.shape[0], 1, trainSC_lamb.shape[1], trainSC_lamb.shape[2])
    trainSC_lamb = trainSC_lamb.view(trainSC_lamb.shape[0], 1, trainSC_lamb.shape[1], trainSC_lamb.shape[2])
    loss = 10
    best_loss = 10
    attempt = 0

    minLoss = 100
    while loss > 0.7 and attempt < 10:
        attempt = attempt + 1
        # print('train net, attempt, currentlose:',attempt,loss)
        net = spectralNet(K=K, num_hidden_layer=layer, activation=activation)
        # best_net = net
        if torch.cuda.is_available():
            # print('Using GPU~~~~~~~~~~~~~~~')
            net.cuda()
        else:
            print('Using CPU  ~~~~~~~~~~~~~~~')
        ## In[] #########  Train the network ################
        net, loss = trainNet(net, trainSC_lamb, trainSC_u, trainFC,trainSC_D,beta=beta,
                             maxIters=2001, lr=1e-3, useCorrLoss=True, FC_u=None)
        if best_loss>loss:
            best_net = net
            best_loss = loss
    return best_net

# In[]
# from a_net0816_corr_tune import spectralNet
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    # device = 'cuda:1'
folder = '../SC_FC_dataset_0905/'


# files = ['SC_RESTINGSTATE_correlationFC','SC_RESTINGSTATE_partialcorrelationFC_L2','SC_EMOTION_correlationFC', 'SC_EMOTION_partialCorrelationFC_L1', 'SC_EMOTION_partialCorrelationFC_L2', 'SC_GAMBLING_correlationFC', 'SC_GAMBLING_partialCorrelationFC_L1', 'SC_GAMBLING_partialCorrelationFC_L2', 'SC_LANGUAGE_correlationFC', 'SC_LANGUAGE_partialCorrelationFC_L1', 'SC_LANGUAGE_partialCorrelationFC_L2', 'SC_MOTOR_correlationFC', 'SC_MOTOR_partialCorrelationFC_L1', 'SC_MOTOR_partialCorrelationFC_L2', 'SC_RELATIONAL_correlationFC', 'SC_RELATIONAL_partialCorrelationFC_L1', 'SC_RELATIONAL_partialCorrelationFC_L2', 'SC_SOCIAL_correlationFC', 'SC_SOCIAL_partialCorrelationFC_L1', 'SC_SOCIAL_partialCorrelationFC_L2', 'SC_WM_correlationFC', 'SC_WM_partialCorrelationFC_L1', 'SC_WM_partialCorrelationFC_L2']
# files = ['SC_EMOTION_partialCorrelationFC_L1', 'SC_EMOTION_partialCorrelationFC_L2', 'SC_GAMBLING_correlationFC', 'SC_GAMBLING_partialCorrelationFC_L1', 'SC_GAMBLING_partialCorrelationFC_L2', 'SC_LANGUAGE_correlationFC', 'SC_LANGUAGE_partialCorrelationFC_L1', 'SC_LANGUAGE_partialCorrelationFC_L2', 'SC_MOTOR_correlationFC', 'SC_MOTOR_partialCorrelationFC_L1', 'SC_MOTOR_partialCorrelationFC_L2', 'SC_RELATIONAL_correlationFC', 'SC_RELATIONAL_partialCorrelationFC_L1', 'SC_RELATIONAL_partialCorrelationFC_L2', 'SC_SOCIAL_correlationFC', 'SC_SOCIAL_partialCorrelationFC_L1', 'SC_SOCIAL_partialCorrelationFC_L2', 'SC_WM_correlationFC', 'SC_WM_partialCorrelationFC_L1', 'SC_WM_partialCorrelationFC_L2']
files = ['SC_RESTINGSTATE_correlationFC']
beta=0
# files = [files[0]]

for file in files:
    data = np.load(folder+file+'.npz')
    # rawSC = torch.from_numpy(data['sc_A'])
    try:
        rawSC = torch.from_numpy(data['sc']).float().to(device)
        rawFC = torch.from_numpy(data['fc']).float().to(device)
    except:
        print("rfMRI dataset!!")
        rawSC = torch.from_numpy(data['rawSC']).float().to(device)
        rawFC = torch.from_numpy(data['rawFC']).float().to(device)

    # allSC = rawSC
    # allFC = rawFC


    allSC = torch.zeros(rawSC.shape, dtype=torch.float32, device=device)
    allFC = torch.zeros(rawFC.shape, dtype=torch.float32, device=device)
    for i in range(rawFC.shape[0]):
        allSC[i] = getLaplacian(rawSC[i],normalized=True)
        allFC[i] = getLaplacian(rawFC[i],normalized=True)

# In[find the best hyperparameters]
    print('training bestnet on validation set')
    entire_size = rawSC.shape[0]

    validation_size = int(entire_size / 5)
    validation_index = list(range(validation_size))
    validation_dataset_SC = allSC[validation_index]
    validation_dataset_FC = allFC[validation_index]
    # validation_train_index = list(range(160,800))
    # validation_test_index = list(range(160))
    validation_train_index = list(range(int(validation_size / 5), validation_size))
    validation_test_index = list(range(int(validation_size / 5)))

    # print("validation_train_index:", validation_train_index, "\num_of_nodes validation_test_index:", validation_test_index)
    sc_validation_train, sc_validation_test = validation_dataset_SC[validation_train_index], validation_dataset_SC[
        validation_test_index]
    fc_validation_train, fc_validation_test = validation_dataset_FC[validation_train_index], validation_dataset_FC[
        validation_test_index]
    bestnet = train_best_spectralNet(sc_validation_train, sc_validation_test,
                                     fc_validation_train, fc_validation_test,
                                     type='N')
    # print('Optimal hyperparameters:', bestnet.K, bestnet.num_hidden_layer, bestnet.activation)

# In[train and evaluate using 5-folder cross-validation]
    dataset_SC = allSC[list(range(validation_size,entire_size))]
    dataset_FC = allFC[list(range(validation_size,entire_size))]
    cvFolers = 5
    kf = KFold(n_splits=cvFolers)
    pearsonCorrelation = np.zeros(cvFolers)
    # kf.get_n_splits(allSC)
    # print(kf)
    cv_idx = 0
    for train_index, test_index in kf.split(dataset_SC):

        # print("TRAIN:", train_index, "\num_of_nodes TEST:", test_index)
        sc_train, sc_test = dataset_SC[train_index], dataset_SC[test_index]
        fc_train, fc_test = dataset_FC[train_index], dataset_FC[test_index]
        kf_validation = KFold(n_splits=cvFolers)

        net = train_spectralNet(sc_train, fc_train,K=bestnet.K,layer=bestnet.num_hidden_layer,activation=bestnet.activation)
        # net = train_spectralNet(sc_train, fc_train,K=5,layer=2,activation=1)

        #####################  Test ########################
        testSC, testSC_lamb, testSC_u, testFC, testFC_lamb, testFC_u, testSC_D = preprocess(sc_test, fc_test)
        testSC_D = testSC_lamb.view(testSC_lamb.shape[0], 1, testSC_lamb.shape[1], testSC_lamb.shape[2])
        testSC_lamb = testSC_lamb.view(testSC_lamb.shape[0], 1, testSC_lamb.shape[1], testSC_lamb.shape[2])

        predFC, FC_vec = net(testSC_lamb,testSC_u,testSC_D,beta=beta)
        evaluation_input_file = folder + 'prediction1107_beta0_' + file

######################### record the results #######################
        correlations, diff = evaluate(predFC, testFC)
        predfc, empiricalfc = tonp(predFC, testFC)
        print('evaluation_input_file:',evaluation_input_file)
        np.savez_compressed(evaluation_input_file, predfc,empiricalfc)

        pearsonCorrelation[cv_idx] = np.mean(correlations)
        print(file+' '+'{:.2f}'.format(pearsonCorrelation[cv_idx]))
        cv_idx = cv_idx+1
        break


    # outfile = folder + 'result1107_beta0_' + file
    # # print(outfile,np.mean(pearsonCorrelation))
    # outtxt = outfile + ' ' + '{:.4f}'.format(np.mean(pearsonCorrelation))+' {:.4f}'.format(np.std(pearsonCorrelation))
    # print(outtxt)
    # file1 = open(folder + 'result1107_beta0.txt', "a")
    # file1.write(outtxt + '\n')
    # file1.close()
    # np.save(outfile, pearsonCorrelation)

