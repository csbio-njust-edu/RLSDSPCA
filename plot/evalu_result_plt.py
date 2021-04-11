import warnings
warnings.filterwarnings("ignore")
from mpl_toolkits.mplot3d import *
from matplotlib import cm

import mpl_toolkits.mplot3d as p3d


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
warnings.filterwarnings("ignore")

labels = ['ACC', 'Macro-REC', 'Macro-PRE', 'Macro-F1', 'Macro-AUC']
erKNN_path = 'evaluation_results.csv'
erKNN = pd.read_csv(erKNN_path)
print(erKNN)
PCA_erKNN =  erKNN.iloc[:,1].tolist()
gLPCA_erKNN =  erKNN.iloc[:,2].tolist()
gLSPCA_erKNN =  erKNN.iloc[:,3].tolist()
RgLPCA_erKNN =  erKNN.iloc[:,4].tolist()
SDSPCA_erKNN =  erKNN.iloc[:,5].tolist()
RLSDSPCA_erKNN =  erKNN.iloc[:,6].tolist()
x = np.arange(len(labels))  # the label locations
width = 0.1  # the width of the bars

plt.rc('font',family='Arial')
fig, ax = plt.subplots()
rects1 = ax.bar(x - width * 5/2, PCA_erKNN, width, label='PCA')
rects2 = ax.bar(x - width * 3/2, gLPCA_erKNN, width, label='gLPCA')
rects3 = ax.bar(x - width/2, gLSPCA_erKNN, width, label='gLSPCA')
rects4 = ax.bar(x + width/2, RgLPCA_erKNN, width, label='RgLPCA')
rects5 = ax.bar(x + width * 3/2, SDSPCA_erKNN, width, label='SDSPCA')
rects6 = ax.bar(x + width * 5/2, RLSDSPCA_erKNN, width, label='RLSDSPCA')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('The average of five evaluation criteria')
ax.set_ylabel('Scores')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper right',ncol=3,borderaxespad = 0.1)
plt.ylim(0.5,1)
plt.show()
