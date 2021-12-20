# In[]
from scipy import io as sio
import numpy as np
folder = '../FCscript/FC_tfMRI_0823/'
# subjects = np.loadtxt('../FCscript/subjects.txt',delimiter='|')


scFile = np.load('original_sc_fc_corr.npz')
scData = scFile['rawSC.npy']

scTxt = np.loadtxt('../original_sc_fc.txt',delimiter=',')

# In[]
import h5py
tasks = ['EMOTION','GAMBLING','LANGUAGE','MOTOR','RELATIONAL','SOCIAL','WM']
fc_types = ['correlationFC','partialCorrelationFC_L1', 'partialCorrelationFC_L2']
# tasks = ['EMOTION']
# fc_types = ['partialCorrelationFC_L2']
scIds = scTxt[:,0]

fileList = []
for task in tasks:
    for type in fc_types:
        validId = []
        sc = np.zeros(scData.shape)
        fc = np.zeros(scData.shape)
        for sid in range(scIds.size):
            subject = int(scIds[sid])
            try:
                tFile = folder + str(subject)+'_'+task+'.mat'
                # print(tFile)
                fcData = h5py.File(tFile,'r')
                # print(fcData.keys())
                fc[len(validId)] = fcData[type][:]
                sc[len(validId)] = scData[sid]
                validId.append(subject)
            except OSError:
                print('Failed %s,%s,%d' % (task,type,subject))
        subjects = np.asarray(validId)
        sc = sc[:len(validId)]
        fc = fc[:len(validId)]
        output = '../SC_taskFC_dataset_0905/SC_'+task+'_'+type
        fileList.append('SC_'+task+'_'+type)
        # np.savez_compressed(output,sc=sc,fc=fc,subjects=subjects)
        # print(subjects.shape[0],output)
print(fileList)
