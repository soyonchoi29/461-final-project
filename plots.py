import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from labellines import *

# read csv file into pandas dataframe
df = pd.read_csv('simulations - mNPUSim.csv', index_col=[i for i in range(12)])

alexnet = df.loc[df['NN Topo'] == 'alexnet']
resnet50 = df.loc[df['NN Topo'] == 'resnet50']

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
# fig.suptitle('Execution Cycle vs. Test Loss')
# # fig.tight_layout(pad=3.0)
#
# ax1.set_title('AlexNet')
# ax1.plot([1, 2, 4, 8], [0.3478, 2.3032, 5.5792, 6.3293])
# ax1.set_xlabel('n')
# ax1.set_ylabel('Final Epoch Training Loss')
#
# ax2.set_title('ResNet50')
# ax2.plot([1, 2, 4, 8], [0.0745, 0.0711, 1.3020, 4.4269])
# ax2.set_xlabel('n')
# ax2.set_ylabel('Final Epoch Training Loss')
#
# labelLines(ax1.get_lines(), zorder=2.5)
# labelLines(ax2.get_lines(), zorder=2.5)
#
# plt.show()
# exit()


# plot stuff
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
fig.suptitle('Execution Cycle of 5th Layer vs. Final Epoch Training Loss')
# fig.tight_layout(pad=3.0)

ax1.set_title('AlexNet')
ax1.scatter([155551, 127713, 137596, 144515], [0.3478, 2.3032, 5.5792, 6.3293])
for i, txt in enumerate(np.array(alexnet['Intermediate Config'])):
    ax1.annotate(txt, (np.array(alexnet['Execution Cycle'])[i], np.array(alexnet['avg_power'])[i]), fontsize=7)
ax1.set_xlabel('Cycles')
ax1.set_ylabel('Avg DRAM Power')

ax2.set_title('Resnet50')
ax2.scatter(np.array(['Cycles per Layer']), np.array(vgg11['avg_power']))
for i, txt in enumerate(np.array(vgg11['Intermediate Config'])):
    ax2.annotate(txt, (np.array(vgg11['Cycles per Layer'])[i], np.array(vgg11['avg_power'])[i]), fontsize=7)
ax2.set_xlabel('Cycles')
ax2.set_ylabel('Avg DRAM Power')

plt.show()

# plt.title('Time vs. Energy')
# # fig.tight_layout(pad=3.0)
#
# ax1.set_title('AlexNet')
# plt.scatter(np.array(alexnet['Cycles per Layer']), np.array(alexnet['avg_power']))
# for i, txt in enumerate(np.array(alexnet['Intermediate Config'])):
#     plt.annotate(txt, (np.array(alexnet['Cycles per Layer'])[i], np.array(alexnet['avg_power'])[i]), fontsize=7)
# ax2.scatter(np.array(vgg11['Cycles per Layer']), np.array(vgg11['avg_power']))
# for i, txt in enumerate(np.array(vgg11['Intermediate Config'])):
#     ax2.annotate(txt, (np.array(vgg11['Cycles per Layer'])[i], np.array(vgg11['avg_power'])[i]), fontsize=7)
#
# plt.xlabel('Cycles')
# plt.ylabel('Avg DRAM Power')
#
# ax2.set_title('VGG11 (Custom)')
# ax2.scatter(np.array(vgg11['Cycles per Layer']), np.array(vgg11['avg_power']))
# for i, txt in enumerate(np.array(vgg11['Intermediate Config'])):
#     ax2.annotate(txt, (np.array(vgg11['Cycles per Layer'])[i], np.array(vgg11['avg_power'])[i]), fontsize=7)
# ax2.set_xlabel('Cycles')
# ax2.set_ylabel('Avg DRAM Power')
#
# ax3.set_title('GPT2')
# ax3.scatter(np.array(gpt2['Cycles per Layer']), np.array(gpt2['avg_power']))
# for i, txt in enumerate(np.array(gpt2['Intermediate Config'])):
#     ax3.annotate(txt, (np.array(gpt2['Cycles per Layer'])[i], np.array(gpt2['avg_power'])[i]), fontsize=7)
# ax3.set_xlabel('Cycles')
# ax3.set_ylabel('Avg DRAM Power')
