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
datapath1 = 'correct_a_b4.csv'
mean_correct_rate = pd.read_csv(datapath1).values
xaxis = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,210]
yaxis = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,210]
# Xaxis,Yaxis = np.meshgrid(xaxis,yaxis)
Xaxis,Yaxis = np.meshgrid(xaxis,yaxis,indexing='ij')
#Times New Roman
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure()
ax = fig.gca(projection='3d')
plt.rc('font',family='Arial')
surf = ax.plot_surface(Xaxis, Yaxis, mean_correct_rate, cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.set_xlabel('Alpha')
ax.set_ylabel('Beta')

# ax.set_xlabel('Alpha')
# ax.set_ylabel('Gamma')

# ax.set_xlabel('Gamma')
# ax.set_ylabel('Beta')

ax.set_zlabel('ACC',rotation=90)
xticks = [10, 60,110,160, 210]
xticklabels = ['-10', '-5',  0, '5', '10']
yticks = [10, 60,110,160, 210]
yticklabels = ['-10', '-5',  0, '5', '10']
plt.xticks(xticks, xticklabels)
plt.yticks(yticks, yticklabels)
#colorbar
l = 0.9
b = 0.2
w = 0.02
h = 1 - 2 * b
rect = [l,b,w,h]
cbar_ax = fig.add_axes(rect)
fig.colorbar(surf, shrink=0.6, aspect=15,cax=cbar_ax)
ax.set_zlim([0,1.0])
ax.view_init(15,135)
fig = surf.get_figure()
plt.show()