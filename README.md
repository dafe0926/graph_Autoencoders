# graph_Autoencoders

**simpleAutoencoder.py** <br />
This python file creates a neural network, generates a set of random graphs, and trains the neural network to embed the set of graphs. This is done in an unsupervised fashion by utilizing an autoencoder architecture. Upon creating and training the neural net off a set of Erdos Renyi random graphs, the neural net is used to embed Erdos-Renyi random graphs from various different parameter values. 20 graphs from the Erdos Renyi ensemble for parameter values ranging 0 and 1. Notably, for p less than ln(n)/n the graphs are embeded differently than those generated for p > ln(n)/n indicating the autoencoder learned a deterministic function to determine whether a graph generated from the Erdos Renyi ensemble was connected or disconnected. 

No data is provided for this file. Simply run the file to recreate the results.

**eigenvectorGraphAutoencoder** <br />

