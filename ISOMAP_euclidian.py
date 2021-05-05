# -*- coding: utf-8 -*-
"""
# =============================================================================
Created on Tue May 26 06:54:37 2020
# =============================================================================
"""
import numpy as np
import math  
from scipy.io import loadmat
from scipy.spatial import distance
from os.path import abspath, exists
import networkx as nx
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %% import data "note to read in the .mat file using loadmat"

def import_data(x):
    # read the graph from 'play_graph.txt'
    f_path = abspath(x)
    if exists(f_path):
         file = loadmat(x)
           
    return file

data = import_data("data/isomap.mat")['images'].T
num_images, num_pixels = data.shape

# %%
def display_image(v):
    n = np.sqrt(len(v)).astype(int)
    img = np.reshape(v, (n, n), order = 'F')
    img = Image.fromarray((img * 255).astype(np.uint8))
    return img
    
r = display_image(data[0,:])
imshow(r,cmap='gray')

# %% calculate weighted adjacency map

w = distance.cdist(data, data, 'euclidean')

# filter adjecency to closest neighbors ensuring each image has at least 100
def wfilter(x, neighbors =100):
    c = x.copy()
    xMin = np.min(np.min(c[c>0]))
    epsilon = xMin
    
    count = 0
    while count <neighbors:
        d = np.where(c>epsilon,0,c)
        count = np.min(np.count_nonzero(d,axis=0))
        epsilon =   epsilon * 1.002
    return(d,epsilon)

    
adjMat,epsilon = wfilter(w,100)


fig = plt.figure()
fig.set_size_inches(10, 8)
htmap_fig = sns.heatmap(adjMat, cmap=sns.color_palette("muted"), cbar_kws={'label': 'Euclidean Distance'})
htmap_fig.axes.set_title("Heat Map of Similarity Graph",fontsize=20)
#plt.savefig("pics/Similarity_Matrix_HeatMap.png")

# %% create and plot graph of adj matrix with some picture on it

# makes dictionary of n random images
def piclabels(n=6):
    n = n
    plotidx = np.random.randint(0,len(data),n)
    labels = {}
    pics= {}
    for i in plotidx:
        pics[i] = display_image(data[i,:])
        labels[i] =str(i)
    return(pics,labels)

pics,labels = piclabels()


# make graph from adj matrix and record positions
G = nx.from_numpy_matrix(adjMat)  
pos=nx.spring_layout(G)

color_map = []
size_map = []
pos2 = {}
for node in G:
    if node in labels.keys():
        color_map.append('red')
        size_map.append(200)
        pos2[node]=pos[node]
    else: 
        color_map.append('blue')
        size_map.append(100)




# draw graph
fig1=plt.figure(figsize=(10, 8))
nx.draw(G, pos, width=.01,edge_color = 'gray', node_color=color_map,node_size=size_map)
nx.draw_networkx_labels(G,pos,labels,font_size=24,font_color='black', font_weight = 'bold')
#plt.savefig("pics/graph_plot_part1_highlighted_nodes.png")


fig2=plt.figure(figsize=(10, 8))
columns = 6
rows = 1
i = 1
ax = []
for node,img in pics.items():
    print(i)
    ax.append(fig2.add_subplot(rows, columns, i))
    ax[-1].set_title(node)  # set title
    plt.imshow(img, cmap ='gray')
    plt.axis('off')
    i= i+1
plt.show()
#plt.savefig("pics/faces_plot_part1_highlighted_nodes.png")


# %% calc graph shortest path
from sklearn.utils.graph_shortest_path import graph_shortest_path
D = graph_shortest_path(adjMat, directed=False, method = 'auto') 



# %%

# make centering matrix
m = len(data)
I = np.identity(m)
one = np.ones(m, dtype=int)
H = I - 1/m*one*one.T


C = (-1/(2))*H@(D**2)@H


#%% Eigan value decomposition

from numpy import linalg as LA

w, v = LA.eig(C)

nPCA =2

eigenValues =  w[0:nPCA]
eiganVectors = v[:,0:nPCA]




dim1 = eiganVectors[:,0]*np.sqrt(eigenValues[0])
dim2 = eiganVectors[:,1]*np.sqrt(eigenValues[1])


fig = plt.figure()
fig.set_size_inches(10, 8)
plt.scatter(dim1,dim2 , s =10)
plt.title('2D Components Using Isomap', fontsize=25)
plt.xlabel('Component: 1', fontsize=20)
plt.ylabel('Component: 2', fontsize=20)

df = {'dim1':dim1,'dim2':dim2}
df = pd.DataFrame(df)

# pca1 left right
df_lr = df.sort_values(by=['dim1'])
image_index_lr = df_lr.rename_axis('image').reset_index()
image_idx_right  = list(image_index_lr['image'])[0:5]
image_idx_left= list(image_index_lr['image'])[-5:]
plt.scatter(df_lr['dim1'][0:5],df_lr['dim2'][0:5] , s =50, c = 'red')
plt.scatter(df_lr['dim1'][-5:],df_lr['dim2'][-5:] , s =50, c = 'green')


# pca2  top and bottom 5
df_tb = df.sort_values(by=['dim2'])
image_index_tb = df_tb.rename_axis('image').reset_index()
image_idx_bottom  = list(image_index_tb['image'])[0:5]
image_idx_top = list(image_index_tb['image'])[-5:]
plt.scatter(df_tb['dim1'][0:5],df_tb['dim2'][0:5] , s =50, c = 'orange')
plt.scatter(df_tb['dim1'][-5:],df_tb['dim2'][-5:] , s =50, c = 'purple')


color_pal = ['red','green','orange','purple']
color_n = [0,1,2,3]
num = 5
color_n = sorted(color_n*num)




def piclabels(image_idx):
    labels = {}
    pics= {}
    
    for i in image_idx:
        pics[i] = display_image(data[i,:])
        labels[i] =str(i)
    return(pics,labels)

img_index = image_idx_left+image_idx_right+image_idx_bottom+image_idx_top

pics,labels = piclabels(img_index)


fig2=plt.figure(figsize=(10, 8))
columns = 5
rows = 4
i = 1
ax = []
j = 0

for node,img in pics.items():
    print(node)
    ax.append(fig2.add_subplot(rows, columns, i))
    ax[-1].set_title(node,color = color_pal[color_n[j]])  # set title
    plt.imshow(img, cmap ='gray')
    plt.axis('off')
    i= i+1
    j= j+1

# %%
df = pd.DataFrame(data)
pixels_per_dimension =64
for idx in df.index:
    df.loc[idx] = df.loc[idx].values.reshape(pixels_per_dimension, pixels_per_dimension).T.reshape(-1)
  
fig = plt.figure()
fig.set_size_inches(10, 10)
ax = fig.add_subplot(111)
ax.set_title('2D Components from Isomap of Facial Images')
ax.set_xlabel('Component: 1')
ax.set_ylabel('Component: 2')

# Show 40 of the images ont the plot
x_size = (max(df_tb['dim1']) - min(df_tb['dim1'])) * 0.08
y_size = (max(df_tb['dim2']) - min(df_tb['dim2'])) * 0.08
for i in range(20):
    img_num = np.random.randint(0, num_images)
    x0 =df_tb.loc[img_num, 'dim1'] - (x_size / 2.)
    y0 = df_tb.loc[img_num, 'dim2'] - (y_size / 2.)
    x1 = df_tb.loc[img_num, 'dim1'] + (x_size / 2.)
    y1 =df_tb.loc[img_num, 'dim2'] + (y_size / 2.)
    img = df.iloc[img_num,:].values.reshape(64, 64)
    ax.imshow(img, aspect='auto', cmap=plt.cm.gray, 
              interpolation='nearest', zorder=100000, extent=(x0, x1, y0, y1))

# Show 2D components plot
ax.scatter(df_tb['dim1'], df_tb['dim2'], marker='.',alpha=0.7)

ax.set_ylabel('Up-Down Pose')
ax.set_xlabel('Right-Left Pose')

plt.show()