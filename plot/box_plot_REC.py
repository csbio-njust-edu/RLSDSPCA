import warnings
warnings.filterwarnings("ignore")
from mpl_toolkits.mplot3d import *
from matplotlib import cm
import mpl_toolkits.mplot3d as p3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
warnings.filterwarnings("ignore")

ALLpath = 'recall_ALL.csv'
ALL = pd.read_csv(ALLpath,header = None)
ALL_er = ALL.iloc[:,[6,5,4,3,2,1]]
print(ALL_er)
plt.rc('font', family='Arial')
plt.figure()
sns.set(style="whitegrid")
plot = sns.boxplot(data= ALL_er, width=0.7)
yrange = np.arange(0,1.1,0.05)
plt.yticks(yrange)
plt.ylim(0.3,1)
plt.xlabel('Methods')
plt.ylabel('Macro-REC')
fig = plot.get_figure()
plt.show()
