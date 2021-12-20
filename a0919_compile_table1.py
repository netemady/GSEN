# In[]
import numpy as np

dataset_method = np.loadtxt('results0917_synthetic.txt', delimiter='&')  # row: dataset, column: method
raw_result = dataset_method.transpose()  # row: method, column: dataset

avg = np.mean(raw_result,axis=1).reshape(raw_result.shape[0],1)
result= np.concatenate((raw_result,avg),axis=1)

np.savetxt('paper_result_syn.txt',result,delimiter='&',fmt='%2.2f')