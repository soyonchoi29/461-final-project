import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# read csv file into pandas dataframe
df = pd.read_csv('simulations - mNPUSim.csv', index_col=[i for i in range(12)])

alexnet = df.loc[df['NN Topo'] == 'alexnet']
resnet50 = df.loc[df['NN Topo']]
gpt2 = df.loc[df['NN Topo'] == 'gpt2']
deepspeech2 = df.loc[df['NN Topo']]

# plot stuff
fig, axes = plt.subplots(2, 2)
fig.suptitle('Time vs. Energy')
fig.tight_layout(pad=3.0)

axes[0,0].set_title('AlexNet')
axes[0,0].scatter(np.array(alexnet['Cycles per Layer']), np.array(alexnet['avg_power']))
for i, txt in enumerate(np.array(alexnet['Intermediate Config'])):
    axes[0,0].annotate(txt, (np.array(alexnet['Cycles per Layer'])[i], np.array(alexnet['avg_power'])[i]), fontsize=7)
axes[0,0].set_xlabel('Cycles')
axes[0,0].set_ylabel('Avg DRAM Power')

# axes[1,0].set_title('Resnet50')
# axes[0,0].plot(np.array(resnet50['Cycles per Layer'], resnet50['Memory Footprint']))
# axes[0,1].set_title('DeepSpeech2')
# axes[0,0].plot(np.array(deepspeech2['Cycles per Layer'], deepspeech2['Memory Footprint']))
# axes[1,1].set_title('GPT2')
# axes[0,0].plot(np.array(gpt2['Cycles per Layer'], gpt2['Memory Footprint']))
plt.show()

