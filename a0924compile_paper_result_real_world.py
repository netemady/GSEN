import numpy as np
dataset_method = np.loadtxt('paper_result_real_world.txt',delimiter='&')
raw_result = dataset_method.transpose()  # row: method, column: dataset
avg = np.mean(raw_result,axis=1).reshape(raw_result.shape[0],1)
result= np.concatenate((raw_result,avg),axis=1)
np.savetxt('paper_result_realworld.txt',result,delimiter='&',fmt='%2.2f')