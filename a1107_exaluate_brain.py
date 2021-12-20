import numpy as np
from a1107_evaluate_func import evaluate_all
RMSE = np.zeros((8, 5))
R2 = np.zeros((8,5))
Pear = np.zeros((8,5))
Spear = np.zeros((8,5))


# # In[]
#
# input_file = 'predictions/mse_prediction_real_IOT20.npz'
# input_data = np.load(input_file)
# real = input_data['arr_0']
# label = input_data['arr_1']
# evaluate_all(label, real)
#
# input_file = 'predictions/mse_prediction_real_IOT40.npz'
# input_data = np.load(input_file)
# real = input_data['arr_0']
# label = input_data['arr_1']
# evaluate_all(label, real)
#
# input_file = 'predictions/mse_prediction_real_IOT60.npz'
# input_data = np.load(input_file)
# real = input_data['arr_0']
# label = input_data['arr_1']
# evaluate_all(label, real)
# print()
#
# input_file = 'predictions/new_prediction_real_IOT20.npz'
# input_data = np.load(input_file)
# real = input_data['arr_0']
# label = input_data['arr_1']
# evaluate_all(label, real)
#
# input_file = 'predictions/new_prediction_real_IOT40.npz'
# input_data = np.load(input_file)
# real = input_data['arr_0']
# label = input_data['arr_1']
# evaluate_all(label, real)
#
# input_file = 'predictions/new_prediction_real_IOT60.npz'
# input_data = np.load(input_file)
# real = input_data['arr_0']
# label = input_data['arr_1']
# evaluate_all(label, real)
#
# print()
# In[Ours]
input_file = '../SC_FC_dataset_0905/prediction1107_beta0_SC_RESTINGSTATE_correlationFC.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[7, 3], R2[7, 3], Pear[7, 3], Spear[7, 3] = evaluate_all(label, real)

input_file = 'predictions/mse_prediction_real_IOT20.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[7, 0], R2[7, 0], Pear[7, 0], Spear[7, 0] = evaluate_all(label, real)

input_file = 'predictions/mse_prediction_real_IOT40.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[7, 1], R2[7, 1], Pear[7, 1], Spear[7, 1] = evaluate_all(label, real)


input_file = 'predictions/mse_prediction_real_IOT60.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[7, 2], R2[7, 2], Pear[7, 2], Spear[7, 2] = evaluate_all(label, real)
print()

# In[baseline]
input_file = 'predictions/prediction_baseline_SC_RESTINGSTATE_correlationFC.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[6, 3], R2[6, 3], Pear[6, 3], Spear[6, 3] = evaluate_all(label, real)

input_file = 'predictions/prediction_baseline_real_IOT20.npz'
input_data = np.load(input_file)
real = -input_data['arr_0']
label = input_data['arr_1']
RMSE[6, 0], R2[6, 0], Pear[6, 0], Spear[6, 0] = evaluate_all(label, real)

input_file = 'predictions/prediction_baseline_real_IOT40.npz'
input_data = np.load(input_file)
real = -input_data['arr_0']
label = input_data['arr_1']
RMSE[6, 1], R2[6, 1], Pear[6, 1], Spear[6, 1] = evaluate_all(label, real)

input_file = 'predictions/prediction_baseline_real_IOT60.npz'
input_data = np.load(input_file)
real = -input_data['arr_0']
label = input_data['arr_1']
RMSE[6, 2], R2[6, 2], Pear[6, 2], Spear[6, 2] = evaluate_all(label, real)
print()

# In[paper 2008]
input_file = 'predictions/prediction_paper2008_brain.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[0, 3], R2[0, 3], Pear[0, 3], Spear[0, 3] = evaluate_all(label, real)

input_file = 'predictions/prediction_paper2008_real_IOT20.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[0, 0], R2[0, 0], Pear[0, 0], Spear[0, 0] = evaluate_all(label, real)

input_file = 'predictions/prediction_paper2008_real_IOT40.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[0, 1], R2[0, 1], Pear[0, 1], Spear[0, 1] = evaluate_all(label, real)

input_file = 'predictions/prediction_paper2008_real_IOT60.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[0, 2], R2[0, 2], Pear[0, 2], Spear[0, 2] = evaluate_all(label, real)
print()

# In[paper 2014]
input_file = 'predictions/prediction_paper2014_brain.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[1, 3], R2[1, 3], Pear[1, 3], Spear[1, 3] = evaluate_all(label, real)

input_file = 'predictions/prediction_paper2014_real_IOT20.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[1, 0], R2[1, 0], Pear[1, 0], Spear[1, 0] = evaluate_all(label, real)

input_file = 'predictions/prediction_paper2014_real_IOT40.npz'
input_data = np.load(input_file)
real = -input_data['arr_0']
label = input_data['arr_1']
RMSE[1, 1], R2[1, 1], Pear[1, 1], Spear[1, 1] = evaluate_all(label, real)

input_file = 'predictions/prediction_paper2014_real_IOT60.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[1, 2], R2[1, 2], Pear[1, 2], Spear[1, 2] = evaluate_all(label, real)
print()

# In[paper 2016]
input_file = 'predictions/prediction_paper2016_brain.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[2, 3], R2[2, 3], Pear[2, 3], Spear[2, 3] = evaluate_all(label, real)

input_file = 'predictions/prediction_paper2016_real_IOT20.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[2, 0], R2[2, 0], Pear[2, 0], Spear[2, 0] = evaluate_all(label, real)

input_file = 'predictions/prediction_paper2016_real_IOT40.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[2, 1], R2[2, 1], Pear[2, 1], Spear[2, 1] = evaluate_all(label, real)

input_file = 'predictions/prediction_paper2016_real_IOT60.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[2, 2], R2[2, 2], Pear[2, 2], Spear[2, 2] = evaluate_all(label, real)
print()
# In[paper 2018]
input_file = 'predictions/prediction_paper2018_brain.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[3, 3], R2[3, 3], Pear[3, 3], Spear[3, 3] = evaluate_all(label, real)

input_file = 'predictions/prediction_paper2018_real_IOT20.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[3, 0], R2[3, 0], Pear[3, 0], Spear[3, 0] = evaluate_all(label, real)

input_file = 'predictions/prediction_paper2018_real_IOT40.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[3, 1], R2[3, 1], Pear[3, 1], Spear[3, 1] = evaluate_all(label, real)

input_file = 'predictions/prediction_paper2018_real_IOT60.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[3, 2], R2[3, 2], Pear[3, 2], Spear[3, 2] = evaluate_all(label, real)
print()

# In[GT-GAN]
input_file = 'GTGAN/prediction_GT_GAN_brain.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[4, 3], R2[4, 3], Pear[4, 3], Spear[4, 3] = evaluate_all(label, real)

input_file = 'predictions/predictions_IOTdata_20_GT_GAN.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[4, 0], R2[4, 0], Pear[4, 0], Spear[4, 0] = evaluate_all(label, real)

input_file = 'predictions/predictions_IOTdata_40_GT_GAN.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[4, 1], R2[4, 1], Pear[4, 1], Spear[4, 1] = evaluate_all(label, real)

input_file = 'predictions/predictions_IOTdata_60_GT_GAN.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[4, 2], R2[4, 2], Pear[4, 2], Spear[4, 2] = evaluate_all(label, real)
print()

# In[C-DGT Wrong results]
# input_file = 'predictions/predictions_C_DGT_Resting_brain.npz'
# input_data = np.load(input_file)
# real = input_data['arr_0']
# label = input_data['arr_1']
# evaluate_all(label,real)
#
# input_file = 'predictions/predictions_IOTdata_20_C_DGT.npz'
# input_data = np.load(input_file)
# real = input_data['arr_0']
# label = input_data['arr_1']
# evaluate_all(label,real)
#
# input_file = 'predictions/predictions_IOTdata_40_C_DGT.npz'
# input_data = np.load(input_file)
# real = input_data['arr_0']
# label = input_data['arr_1']
# evaluate_all(label,real)
#
# input_file = 'predictions/predictions_IOTdata_60_C_DGT.npz'
# input_data = np.load(input_file)
# real = input_data['arr_0']
# label = input_data['arr_1']
# evaluate_all(label,real)
# print()

# In[C-DGT1]
input_file = 'predictions/predictions_C_DGT_Resting_brain.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[5, 3], R2[5, 3], Pear[5, 3], Spear[5, 3] = evaluate_all(label, real)

input_file = 'predictions/predictions_IOTdata_20_C_DGT1.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[5, 0], R2[5, 0], Pear[5, 0], Spear[5, 0] = evaluate_all(label, real)

input_file = 'predictions/predictions_IOTdata_40_C_DGT1.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[5, 1], R2[5, 1], Pear[5, 1], Spear[5, 1] = evaluate_all(label, real)

input_file = 'predictions/predictions_IOTdata_60_C_DGT1.npz'
input_data = np.load(input_file)
real = input_data['arr_0']
label = input_data['arr_1']
RMSE[5, 2], R2[5, 2], Pear[5, 2], Spear[5, 2] = evaluate_all(label, real)
print()


# In[]
import pandas as pd
def toLatex(arr):
    df = pd.DataFrame(np.round(arr,4), index=['Galan2008','Abdelnour2014','Meier2016','Abdelnour2018','GT-GAN','C-DGT','Baseline','Ours'])
    print(df.to_latex())

RMSE[:, 4] = np.mean(RMSE[:, :4], axis = 1)
R2[:,4] = np.mean(R2[:,:4],axis = 1)
Pear[:,4] = np.mean(Pear[:,:4],axis = 1)
Spear[:,4] = np.mean(Spear[:,:4],axis = 1)

toLatex(RMSE)
toLatex(R2)
toLatex(Pear)
toLatex(Spear)