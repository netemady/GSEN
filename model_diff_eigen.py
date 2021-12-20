# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 18:17:13 2019
@author: gxjco
@modified by Qingzhe
this code is for diffusion model in paper:
    ''Abdelnour F, Voss HU, Raj A.
    Network diffusion accurately models the relationship between structural and functional brain connectivity networks. Neuroimage.
     2014 Apr 15;90:335-47.''

and eigen model in paper:
    ''Abdelnour F, Dayan M, Devinsky O, Thesen T, Raj A.
    Functional brain connectivity is predictable from anatomic network's Laplacian eigen-structure. NeuroImage.
     2018 May 15;172:728-39.''
"""
import scipy
from scipy.stats import pearsonr
import numpy as np
from helper import *
# In[]
allData = np.loadtxt('original_sc_fc.txt',delimiter=',')
# In[]
# sc_mat=np.load('sc.npy')
# fc_mat=np.load('fc.npy')
sizeNode = 68
sizeConnectivity = sizeNode*sizeNode
row_count = allData.shape[0]

#compute laplacian
sc_mat = np.zeros((row_count,sizeNode,sizeNode))
fc_mat = np.zeros((row_count,sizeNode,sizeNode))
for row in range(row_count):
        sc_mat[row,:,:] = allData[row, 1:sizeConnectivity+1].reshape((sizeNode, sizeNode))
        fc_mat[row,:,:] = allData[row, sizeConnectivity+1:2*sizeConnectivity+1].reshape((sizeNode, sizeNode))
# In[]
def laplacian(mat):
         rowsum=np.sum(mat,0)
         d_inv_sqrt =np.power(rowsum,-0.5)
         lamda = np.diag(d_inv_sqrt)
         I=np.eye(mat.shape[1])
         laplacian = I -np.dot(np.dot(lamda,mat),lamda)
         return laplacian
     
def predict_fc_diff(sc,beta_t):
    sc_lap=laplacian(sc)
    #a,b=np.linalg.eig(sc_lap)
    #a[0]=0
    #a[1]=0
    #sc_lap=np.dot(np.dot(a,b),np.transpose(b))
    fc=np.exp(-1*beta_t*sc_lap)
    return fc

def predict_fc_eigen(sc,a,beta_t,b):
    sc_lap=laplacian(sc)
    #a,b=np.linalg.eig(sc_lap)
    #a[0]=0
    #a[1]=0
    #sc_lap=np.dot(np.dot(np.diag(a),b),np.transpose(b))       
    fc=a*np.exp(-beta_t*sc_lap)+b*np.eye(sc.shape[1])
    return fc


#fitting the paramters
def curve_fiting_eigen(sc_train,fc_train):
  def func(x,a,b,c):
      return a * np.exp(-b * x) + c
  from scipy.optimize import curve_fit
  popt, pcov = curve_fit(func, sc_train, fc_train)
  return popt

def curve_fiting_diff(sc_train,fc_train):
  def func(x,b):
      return np.exp(-b * x)
  from scipy.optimize import curve_fit
  popt, pcov = curve_fit(func, sc_train, fc_train)
  return popt

sc_eigenvalues=np.zeros((sc_mat.shape[0],sc_mat.shape[1]))
fc_eigenvalues=np.zeros((fc_mat.shape[0],fc_mat.shape[1]))
for i in range(len(sc_mat)):
   sc_eigenvalues[i]=np.linalg.eig(laplacian(sc_mat[i]))[0]
   fc_eigenvalues[i]=np.linalg.eig(fc_mat[i])[0]

popt_eigen=curve_fiting_eigen(np.reshape(sc_eigenvalues,[-1])[2:],np.reshape(fc_eigenvalues,[-1])[2:])
popt_diff=curve_fiting_diff(np.reshape(sc_eigenvalues,[-1])[2:],np.reshape(fc_eigenvalues,[-1])[2:])



#predict fc
pred_fc_diff=np.zeros((fc_mat.shape[0],fc_mat.shape[1],fc_mat.shape[2]))
pred_fc_eigen=np.zeros((fc_mat.shape[0],fc_mat.shape[1],fc_mat.shape[2]))
for i in range(len(sc_mat)):
    pred_fc_diff[i]=predict_fc_diff(sc_mat[i],popt_diff[0])
    pred_fc_eigen[i]=predict_fc_eigen(sc_mat[i],popt_eigen[0],popt_eigen[1],popt_eigen[2])

# In[]
# #evaluate
def pearson_corr(fc_mat,pred_fc):
    norm=np.ones((fc_mat.shape[1],fc_mat.shape[2]))-np.eye(fc_mat.shape[1])
    r=[]
    p=[]
    for i in range(len(fc_mat)):
          r.append(pearsonr(np.reshape(np.multiply(fc_mat[i],norm),-1), np.reshape(np.multiply(pred_fc[i],norm),-1))[0])
          p.append(pearsonr(np.reshape(np.multiply(fc_mat[i],norm),-1), np.reshape(np.multiply(pred_fc[i],norm),-1))[1])
    max_r=np.max(r)
    max_p=np.max(p)
    min_r=np.min(r)
    min_p=np.min(p)
    mean_r=np.mean(r)
    mean_p=np.mean(p)
    var_r=np.var(r)
    var_p=np.var(p)
    return max_r,min_r,mean_r,var_r, max_p,min_p,mean_p,var_p
max_r,min_r,mean_r,var_r, max_p,min_p,mean_p,var_p=pearson_corr(fc_mat,sc_mat)

