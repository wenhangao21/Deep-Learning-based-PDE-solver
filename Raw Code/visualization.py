import pandas as pd
import numpy as np
import scipy.io as sio
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import imageio
import os
from PIL import Image
import sys


plt.rcParams['font.sans-serif'] = ['SimHei']  # Set character format to avoid garbled code
plt.rcParams['axes.unicode_minus'] = False

data1 = loadmat("grid_points.mat")    # Import data from grid_points.mat
data2 = loadmat("net_solution.mat")   # Import data from net_solution.mat
data3 = loadmat("true_solution.mat")  # Import data from true_solution.mat

data01 = data1['grid_points']
data02 = data2['net_solution'].reshape(95305, 1)     #reshape to 95305rows 1colunm
data03 = data3['true_solution'].reshape(95305, 1)

df01 = pd.DataFrame(data01, columns=['t', 'x', 'y'])
df02 = pd.DataFrame(data02, columns=['z1(net)'])
df03 = pd.DataFrame(data03, columns=['z2(true)'])
df = pd.concat([df01, df02, df03], axis=1)

df1 = df.set_index(['t'])   # Customize t as index

ids = df1.index   # Remove duplicate indexes
list1 = list(set(ids))
def get_png():                         #define the function of plotting every point
    for i in list1:                # Traverse the t in the list and draw its three-dimensional graph for each t
                                       # Imput data of x y z
        x = df1['x'][i]
        y = df1['y'][i]
        z = df1[new_z][i]  

        fig = plt.figure()
        ax = Axes3D(fig)

        ax.set_title('t={}'.format(i))         # Set title
        ax.scatter(x, y, z, c=z, cmap="cool")  # print xyz as scatter plot

        ax.set_zlim3d(-0.05, 1.66)             # Fixed z-axis length
        ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
        ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
        ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})

        surf = ax.scatter(x, y, z, c=z, cmap="cool", vmin=-0.05, vmax=1.66)

        fig.colorbar(surf, shrink=0.5, aspect=5)  # plot the colorbar with fixed length

        plt.savefig(r'{}.png'.format(i), dpi=200)  # save all picture to document



# make the gif of 'net_solution'
new_z = 'z1(net)'
get_png()
images = []  # set a empty list
# Traverse the folder and fill the images generated above into 'filenames' in order
filenames = sorted((fn for fn in os.listdir('.') if fn.endswith('.png')))
# Generate GIF image
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('network.gif', images, duration=0.3)

# make the gif of 'net_solution'
new_z = 'z2(true)'
get_png()
images = []  # set a empty list
# Traverse the folder and fill the images generated above into 'filenames' in order
filenames = sorted((fn for fn in os.listdir('.') if fn.endswith('.png')))
# Generate GIF image
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('true.gif', images, duration=0.3)


#Delete the .png file after getting the gif file
import os
import glob
 

for infile in glob.glob(os.path.join('*.png')):
     os.remove(infile)
