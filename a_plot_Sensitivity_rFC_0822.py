# In[]
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
# In[]
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 3), dpi=80, facecolor='w', edgecolor='k')
res = np.load('result_original_sc_fc_corr.npz')['result']
data = res[:100,4:5].reshape((10, 10))
ax = plt.subplot(131)
fig = ax.imshow(data, cmap='hot', vmin=0, vmax=1)
# plt.colorbar(fig)

# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(data.shape[0]), minor=False)
ax.set_yticks(np.arange(data.shape[1]), minor=False)
ax.set_xticklabels(np.arange(data.shape[0]) + 1)
ax.set_yticklabels(np.arange(data.shape[0]) + 1)
plt.title('Activation:Sigmoid')
plt.xlabel('layer count')
plt.ylabel('power: K')

data = res[100:200,4:5].reshape((10, 10))
ax = plt.subplot(132)
fig = ax.imshow(data, cmap='hot', vmin=0, vmax=1)

# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(data.shape[0]), minor=False)
ax.set_yticks(np.arange(data.shape[1]), minor=False)
ax.set_xticklabels(np.arange(data.shape[0]) + 1)
ax.set_yticklabels(np.arange(data.shape[0]) + 1)
plt.title('Activation:Relu')
plt.xlabel('layer count')
plt.ylabel('power: K')

data = res[200:300,4:5].reshape((10, 10))
ax = plt.subplot(133)
fig = ax.imshow(data, cmap='hot', vmin=0, vmax=1)

# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(data.shape[0]), minor=False)
ax.set_yticks(np.arange(data.shape[1]), minor=False)
ax.set_xticklabels(np.arange(data.shape[0]) + 1)
ax.set_yticklabels(np.arange(data.shape[0]) + 1)
plt.title('Activation:Tanh')
plt.xlabel('layer count')
plt.ylabel('power: K')

# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(fig, fraction=0.046, pad=0.02)

plt.savefig('sensitivity_rFC_corr.png')
plt.show()

# In[]
res = np.load('result_original_sc_fc_partial.npz')['result']
data = res[:100,4:5].reshape((10, 10))
figure(num=None, figsize=(8, 3), dpi=80, facecolor='w', edgecolor='k')

ax = plt.subplot(131)
fig = ax.imshow(data, cmap='hot', vmin=0, vmax=1)
# plt.colorbar(fig)

# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(data.shape[0]), minor=False)
ax.set_yticks(np.arange(data.shape[1]), minor=False)
ax.set_xticklabels(np.arange(data.shape[0]) + 1)
ax.set_yticklabels(np.arange(data.shape[0]) + 1)
plt.title('Activation:Sigmoid')
plt.xlabel('layer count')
plt.ylabel('power: K')

data = res[100:200,4:5].reshape((10, 10))
ax = plt.subplot(132)
fig = ax.imshow(data, cmap='hot', vmin=0, vmax=1)

# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(data.shape[0]), minor=False)
ax.set_yticks(np.arange(data.shape[1]), minor=False)
ax.set_xticklabels(np.arange(data.shape[0]) + 1)
ax.set_yticklabels(np.arange(data.shape[0]) + 1)
plt.title('Activation:Relu')
plt.xlabel('layer count')
plt.ylabel('power: K')

data = res[200:300,4:5].reshape((10, 10))
ax = plt.subplot(133)
fig = ax.imshow(data, cmap='hot', vmin=0, vmax=1)

# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(data.shape[0]), minor=False)
ax.set_yticks(np.arange(data.shape[1]), minor=False)
ax.set_xticklabels(np.arange(data.shape[0]) + 1)
ax.set_yticklabels(np.arange(data.shape[0]) + 1)
plt.title('Activation:Tanh')
plt.xlabel('layer count')
plt.ylabel('power: K')

# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(fig, fraction=0.046, pad=0.02)

plt.savefig('sensitivity_rFC_partial.png')
plt.show()

# In[]
res = np.load('result_real_IOT20.npz')['result'][:,4:5]
res =np.nan_to_num(res,nan=0)
res[res<0] = 0
data = res[:100,:].reshape((10, 10))
figure(num=None, figsize=(8, 3), dpi=80, facecolor='w', edgecolor='k')

ax = plt.subplot(131)
fig = ax.imshow(data, cmap='hot', vmin=0, vmax=1)
# plt.colorbar(fig)

# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(data.shape[0]), minor=False)
ax.set_yticks(np.arange(data.shape[1]), minor=False)
ax.set_xticklabels(np.arange(data.shape[0]) + 1)
ax.set_yticklabels(np.arange(data.shape[0]) + 1)
plt.title('Activation:Sigmoid')
plt.xlabel('layer count')
plt.ylabel('power: K')

data = res[100:200,:].reshape((10, 10))
ax = plt.subplot(132)
fig = ax.imshow(data, cmap='hot', vmin=0, vmax=1)

# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(data.shape[0]), minor=False)
ax.set_yticks(np.arange(data.shape[1]), minor=False)
ax.set_xticklabels(np.arange(data.shape[0]) + 1)
ax.set_yticklabels(np.arange(data.shape[0]) + 1)
plt.title('Activation:Relu')
plt.xlabel('layer count')
plt.ylabel('power: K')

data = res[200:300,:].reshape((10, 10))
ax = plt.subplot(133)
fig = ax.imshow(data, cmap='hot', vmin=0, vmax=1)

# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(data.shape[0]), minor=False)
ax.set_yticks(np.arange(data.shape[1]), minor=False)
ax.set_xticklabels(np.arange(data.shape[0]) + 1)
ax.set_yticklabels(np.arange(data.shape[0]) + 1)
plt.title('Activation:Tanh')
plt.xlabel('layer count')
plt.ylabel('power: K')

# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(fig, fraction=0.046, pad=0.02)

plt.savefig('sensitivity_IOT20.png')
plt.show()






