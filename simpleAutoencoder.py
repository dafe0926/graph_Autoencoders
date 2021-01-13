# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 14:15:33 2021

@author: fergu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 19:13:47 2019

@author: fergu
"""
from tensorflow.keras import backend
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from networkx import nx
import scipy.io as spio 
import numpy as np
import matplotlib as mpl

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

c1='#1f77b4' #blue
c2='green' #green



np.random.seed(0)
N = 100 #num of graphs to train on
n = 200 #num nodes in the graph
d = 2 #num of dimensions to embed the graphs in


inputs = Input(shape=(n*n,))
# a layer instance is callable on a tensor, and returns a tensor
output_1 = Dense(1000, activation='tanh')(inputs)
output_2 = Dense(250,activation='relu')(output_1)
output_3 = Dense(d, name = 'embeded')(output_2)
output_4 = Dense(250, activation='relu')(output_3)
output_5 = Dense(1000,activation = 'tanh')(output_4)
predictions = Dense(n*n)(output_5)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

layer_name = 'embeded'

intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)


#Generate data for training
data = []
for i in range(0,N):
    G = nx.fast_gnp_random_graph(n, 0.5)
    G_dense = nx.adjacency_matrix(G).todense()
    data.append(np.array(np.matrix.flatten(nx.adjacency_matrix(G).todense()),dtype = np.float64)[0])
data = np.array(data)

model.fit(data, data,epochs = 3)  # starts training


graphs = 20
data = []
numSets = 100
for p in range(1,numSets):
    for g in range(0,graphs):
        G = nx.fast_gnp_random_graph(n, (p)/numSets)
        data.append(np.array(np.matrix.flatten(nx.adjacency_matrix(G).todense()),dtype = np.float64)[0])
data = np.array(data)

fig, ax = plt.subplots()
embededGraphs = intermediate_layer_model.predict(data)

for i in range(0,len(data)):
    ax.scatter(embededGraphs[i][0],embededGraphs[i][1],color=colorFader(c1,c2,i/len(data)))
    
ax.imshow()






