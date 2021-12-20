# Created by qli10 at 8/13/2019

# Created by qli10 at 8/12/2019
# power

# In[load training data]
import numpy as np
import torch
from helper import *
allData = np.loadtxt('original_sc_fc.txt',delimiter=',')
# In[load training data]
useAbs = False
normalized = True

trainSize = 700
testSize = allData.shape[0]-trainSize
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
sizeConnectivity = int((allData.shape[1 ] -1 ) /3)
sizeNode = int(np.sqrt(sizeConnectivity))
triangleSize = int(sizeNode *(sizeNode +1 ) /2)
trainData = torch.from_numpy(allData[:trainSize ,1:]).float()
testData = torch.from_numpy(allData[trainSize: ,1:]).float()
# testSize=trainSize;testData = torch.from_numpy(allData[:testSize ,1:]).float()   # for debug only, overfitting
trainID = allData[:trainSize ,0]
testID = allData[:trainSize: ,0]
trainSC = []
trainSC_u = torch.zeros(trainSize,sizeNode ,sizeNode ,dtype=torch.float32 ,device=device)
trainFC_u = torch.zeros(trainSize,sizeNode ,sizeNode ,dtype=torch.float32 ,device=device)

trainSC_lamb = torch.zeros(trainSize ,sizeNode ,dtype=torch.float32 ,device=device)
# trainFC_corr = []
trainFC_corr = torch.zeros(trainSize,sizeNode ,sizeNode ,dtype=torch.float32 ,device=device)


# trainFC_corr_u = []
trainFC_corr_lamb = torch.zeros(trainSize ,sizeNode ,dtype=torch.float32 ,device=device)
# trainFC_pc = []
# trainFC_pc_u = []
# trainFC_pc_lamb = np.zeros((trainSize,sizeNode))

for row in range(trainSize):
    sc_A = trainData[row, :sizeConnectivity].reshape((sizeNode, sizeNode))
    sc = getLaplacian(sc_A ,normalized=normalized)
    trainSC.append(sc)
    lamb_sc ,u_sc = torch.symeig(sc ,eigenvectors=True)
    trainSC_u[row] = u_sc
    trainSC_lamb[row ,:] = lamb_sc
    if useAbs:
        fc_corr_A = torch.abs(trainData[row, sizeConnectivity: 2 *sizeConnectivity].reshape((sizeNode, sizeNode)))
    else:
        fc_corr_A = (trainData[row, sizeConnectivity: 2 *sizeConnectivity].reshape((sizeNode, sizeNode)))

    L_fc = getLaplacian(fc_corr_A ,normalized=normalized)
    trainFC_corr[row]=L_fc
    lamb_fc, u_fc = torch.symeig(L_fc ,eigenvectors=True)
    trainFC_corr_lamb[row, :] = lamb_fc
    trainFC_u[row] = u_fc


# In[]


# from spectralNet_sigmoid import spectralNet
# Created by qli10 at 8/7/2019

# import numpy as np
import torch.nn as nn
class spectralNet(nn.Module):
# class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

    def __init__(self,K=5,num_hidden_layer=10,debug=False):
        super(spectralNet, self).__init__()
        self.K = K
        self.in_channel_hidden = K
        self.out_channel_hidden = K
        self.num_hidden_layer = num_hidden_layer
        self.debug = debug
        self.conv1 = nn.Sequential(  # input shape (1, 1, 68)
            nn.Conv1d(
                in_channels=1,  # input = 1, e.g., no Tylor expantion
                out_channels=self.K,  # n_filters, e.g, number of Tylor expansion K=16
                kernel_size=1,  # filter size
                stride=1,  # filter movement/step
                padding=0, # no need padding
                bias = True,
            ),  # output shape (16,68,1)
            # nn.Sigmoid(),  # activation
            # nn.ReLU(),  # activation
            # nn.Softplus(),
        )


        self.conv_hidden = nn.Sequential(  # input shape (1, 1, 68)
            nn.Conv1d(
                in_channels=self.in_channel_hidden,  #
                out_channels=self.out_channel_hidden,  #
                # out_channels=1,  #0
                kernel_size=1,  # filter size
                stride=1,  # filter movement/step
                padding=0,  # no need padding
                groups=self.in_channel_hidden,
                bias=True,
            ),  # output shape (16,68,1)
            nn.Sigmoid(),  # activation
            # nn.ReLU(),  # activation
            # nn.Tanh(),
            # nn.Softplus(),
        )
        self.conv_out = nn.Sequential(  # input shape (1, 1, 68)
            nn.Conv1d(
                in_channels=68,  # ROIs
                out_channels=68,  # ROIs
                kernel_size=self.out_channel_hidden,  # filter size
                # kernel_size=1,
                stride=1,  # filter movement/step
                padding=0,  # no need padding
                bias=True,
            ),
            # nn.Sigmoid(),  # activation
            # nn.ReLU(),  # activation
            # nn.Tanh()
        )

        # self.out = nn.Linear(K*68, 68)  # fully connected layer, output 68 values of lambda

    def forward(self, x,u):
        self.u = u
        x = x.transpose(1,2)  #from 700*68*1 to 700*1*68
        x = torch.cat([x**i for i in range(1,self.K+1)],1)
        if self.debug:
            print('input_layer1.shape:',x.shape)
            print('x_input:',x[0,0,:])
            # print('x_input:', x[0, 1, :])
        x = self.conv_hidden(x)
        x = torch.sum(x,1).reshape(x.shape[0],1,x.shape[2])
        x = torch.cat([x**i for i in range(1,self.K+1)],1)
        for layer in range(self.num_hidden_layer):
            x = self.conv_hidden(x)
            x = torch.sum(x, 1).reshape(x.shape[0], 1, x.shape[2])
            x = torch.cat([x ** i for i in range(1, self.K + 1)], 1)
        #

        # # print('out_x.shape:', x.shape)
        # # x = torch.sum(x, 1).reshape(x.shape[0], 1, x.shape[2])
        # # x = x.transpose(2, 1)
        #
        #
        # # use conv
        x = x.transpose(2, 1)
        # print('x0=',x[0],x[0].shape)

        x = self.conv_out(x)

        # x = self.conv2(x)   #
        # print('x=',x[0],x[0].shape)
        x = x.expand(x.shape[0], x.shape[1], x.shape[1])  # copy lambda vector to diag(lambda)
        x = torch.mul(x,torch.eye(x.shape[1]).cuda())
        output = torch.matmul(torch.matmul(self.u,x),self.u.transpose(1,2))
        # print('x.shape:',x.shape)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 68)
        # output = self.out(x)
        # output = x
        # print('output.shape:', output.shape)
        return output, x  # return x for visualization
## In[]
maxIters = 200
net = spectralNet()
if torch.cuda.is_available():
    print('Using GPU~~~~~~~~~~~~~~~')
    net.cuda()
else:
    print('Using CPU  ~~~~~~~~~~~~~~~')

print(net)  # net structure
optimizer = torch.optim.Adam(net.parameters(), lr=1e-1)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# def correlation_loss(x,y):
def DeepCoral(source, target):
    # d = source.data.shape[1]
    # xm = torch.mean(source, 1, keepdim=True)
    # xc = torch.matmul(torch.transpose(xm, 0, 1), xm)  # source covariance
    # xmt = torch.mean(target, 1, keepdim=True)
    # xct = torch.matmul(torch.transpose(xmt, 0, 1), xmt)  # target covariance
    # loss = torch.mean(torch.mul((xc - xct), (xc - xct)))  # frobenius norm between source and target
    # res =  - loss / (4 * d * d)
    # return res
    # x = torch.cat([x ** i for i in range(1, self.K + 1)], 1)
    # print(source.shape,target.shape)
    row_idx, col_idx = torch.triu_indices(source.shape[1], source.shape[2],offset=1)
    x = source[:, row_idx, col_idx]
    y = target[:, row_idx, col_idx]
    # print('triu.shape',x.shape,y.shape)
    #
    #
    # x = torch.cat([source[i][torch.triu(torch.ones(source.shape[1], source.shape[2]), diagonal=1) == 1].reshape(1,2278) for i in range(source.shape[0])],0)
    # y = torch.cat([target[i,torch.triu(torch.ones(target.shape[1], target.shape[2]), diagonal=1) == 1].reshape(1,2278) for i in range(target.shape[0])],0)
    # x = source
    # y = target
    vx = x - torch.mean(x, 1, keepdim=True)
    vy = y - torch.mean(y, 1, keepdim=True)
    xy_cov = torch.bmm(vx.view(vx.shape[0], 1, vx.shape[1]), vy.view(vy.shape[0], vy.shape[1], 1), ).view(vx.shape[0])
    # std_x = torch.std(vx,dim=1)
    # std_y = torch.std(vy,dim=1)
    # xy_std = torch.mul(std_x,std_y)

    # corr = torch.div(xy_cov,xy_std)
    cost = xy_cov / torch.mul(torch.sqrt(torch.sum(vx ** 2, dim=1)), torch.sqrt(torch.sum(vy ** 2, dim=1)))
    # print('train_corr',torch.mean(cost),x.shape,y.shape)
    loss = 1 - torch.mean(cost)
    # cost = torch.sum(vx * vy) ./ (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

    return loss

loss_func = torch.nn.MSELoss()
# loss_func = DeepCoral()
trainSC_lamb_3d = trainSC_lamb.reshape((trainSC_lamb.shape[0],trainSC_lamb.shape[1],1))  #700*68*1
# trainFC_corr_lamb_4d = trainFC_corr_lamb.reshape((trainFC_corr_lamb.shape[0],1,1,trainFC_corr_lamb.shape[1]))
for t in range(maxIters):
    # train_predict_FC_lamb = net(trainSC_lamb_3d)[0]
    # loss = loss_func(train_predict_FC_lamb, trainFC_corr_lamb)

    #### mse loss
    # train_predict_FC = net(trainSC_lamb_3d)[0]
    # loss = loss_func(train_predict_FC, trainFC_corr)
    ####  correlation_loss
    train_predict_FC,fc_vec = net(trainSC_lamb_3d,trainSC_u)
    # train_predict_FC,fc_vec = net(trainSC_lamb_3d,trainFC_u)    ############### debug only
    # loss = DeepCoral(fc_vec, trainFC_corr.view(trainFC_corr.size(0), -1))
    loss = DeepCoral(train_predict_FC, trainFC_corr)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 50 == 0:
        print(loss, t)


## In[load test data]

testSC = []
testSC_u = torch.zeros(testSize, sizeNode, sizeNode, dtype=torch.float32, device=device)
testFC_u = torch.zeros(testSize, sizeNode, sizeNode, dtype=torch.float32, device=device)
testSC_lamb = torch.zeros(testSize,sizeNode,dtype=torch.float32,device=device)
test_triangle_size = int(sizeNode*(sizeNode-1)/2)
# test_FCs = torch.zeros(testSize * test_triangle_size, 1)
# predict_FCs = torch.zeros(int(testSize*sizeNode*(sizeNode-1)/2),1)
emprical_FC = []
# testFC_corr_u = []
# testFC_corr_lamb = torch.zeros((trainSize,sizeNode),dtype=torch.float32)
# trainFC_pc = []
# trainFC_pc_u = []
# trainFC_pc_lamb = np.zeros((trainSize,sizeNode))
for row in range(testSize):
    sc_A = testData[row, :sizeConnectivity].reshape((sizeNode, sizeNode))

    # sc_D = torch.diag(torch.sum(sc_A, dim=0))
    # sc = sc_D - sc_A
    sc = getLaplacian(sc_A, normalized=normalized)
    # sc = sc_A
    testSC.append(sc)
    lamb_sc,u_sc = torch.symeig(sc,eigenvectors=True)
    testSC_u[row] = u_sc
    testSC_lamb[row,:] = lamb_sc
    if useAbs:
        fc_corr_A = torch.abs(testData[row, sizeConnectivity:2 * sizeConnectivity].reshape((sizeNode, sizeNode)))
    else:
        fc_corr_A = (testData[row, sizeConnectivity:2 * sizeConnectivity].reshape((sizeNode, sizeNode)))
    fc_corr_A = getLaplacian(fc_corr_A,normalized=normalized)

    emprical_FC.append(fc_corr_A)
    lamb_fc,u_fc = torch.symeig(fc_corr_A,eigenvectors=True)
    testFC_u[row] = u_fc
## In[test]
# testSC_lamb_3d = testSC_lamb.reshape((testSC_lamb.shape[0],1,testSC_lamb.shape[1]))
testSC_lamb_3d = testSC_lamb.reshape((testSC_lamb.shape[0],testSC_lamb.shape[1],1))

test_predict_FC_laplacian = net(testSC_lamb_3d,testSC_u)[0]
# test_predict_FC_laplacian = net(testSC_lamb_3d,testFC_u)[0]    ############# debug only


## In[evaluate Result]
predict_FCs = torch.zeros(int(testSize*sizeNode*(sizeNode-1)/2),1)
result_pearson = np.zeros(testSize)
diff = np.zeros((testSize,int(67*68/2)))
for row in range(testSize):
    # predict_FC_laplacian  = (torch.mm(testSC_u[row],torch.mm(vec2triu(test_predict_FC_lamb[row,:].t().cpu()),testSC_u[row].t())))

    # predict_FCs[row * test_triangle_size:(row + 1) * test_triangle_size, 0] = -triu2vec(predict_FC_laplacian, diag=1)
    predict_FC_vec = triu2vec(test_predict_FC_laplacian[row].cpu(), diag=1)
    # print('pearson:',row,pearsonCorrelation(emprical_FC[row],predict_FC_vec)
    # ,pearsonr(emprical_FC[row].detach().numpy(),predict_FC_vec.detach().numpy()))
    # resul
    # t_pearson.append(pearsonCorrelation(emprical_FC[row],predict_FC_vec))
    emprical_fc = triu2vec(emprical_FC[row], diag=1).detach().numpy().reshape(1,predict_FC_vec.shape[0])
    predict_fc = predict_FC_vec.detach().numpy().reshape(1,predict_FC_vec.shape[0])
    diff[row,:] = emprical_fc-predict_fc
    # print('diff',diff)
    (pearson, p_val) = pearsonr(emprical_fc.flatten(), predict_fc.flatten())
    result_pearson[row] = pearson


    truefc = emprical_FC[row].detach().flatten().numpy()
    predfc = test_predict_FC_laplacian[row].detach().flatten().cpu().numpy()
    (allpearson, allp_val) = pearsonr(truefc, predfc)
    # print('pearson corr = ',pearson,allpearson)
print('mean pearson:',result_pearson.mean(),result_pearson.std())
# print('all mean pearson:',result_pearson.mean(),result_pearson.std())

## In[]
a = emprical_FC[0].numpy()
b = test_predict_FC_laplacian[0].cpu().detach().numpy()

