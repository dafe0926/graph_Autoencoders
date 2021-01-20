# graph_Autoencoders

**simpleAutoencoder.py** <br />
This python file creates a neural network, generates a set of random graphs, and trains the neural network to embed the set of graphs. This is done in an unsupervised fashion by utilizing an autoencoder architecture. Upon creating and training the neural net off a set of Erdos Renyi random graphs, the neural net is used to embed Erdos-Renyi random graphs from various different parameter values. 20 graphs from the Erdos Renyi ensemble for parameter values ranging 0 and 1. Notably, for p less than ln(n)/n the graphs are embeded differently than those generated for p > ln(n)/n indicating the autoencoder learned a deterministic function to determine whether a graph generated from the Erdos Renyi ensemble was connected or disconnected. <br />

No data is provided for this file. Simply run the file to recreate the results. <br />

**autoencoder_Naive_Aggregation.py** <br />
This python files creates a neural network, generates a few random graphs, and trains the neural entwork to embed the NODES of the graphs. This is done in an unsupervised fashion by utilizing an autoencoder architecture. Upon creating and training the neural net, the embeded nodes of a graph are aggregated by simply summing their embeded values to acheive an embedding of the entire graph. The images display both the embeded graph as well as the embeded set of nodes that are aggregated. Note this second image informs us that the aggregation function used is not necessarily the best, perhaps a PCA analysis of the embeded nodes will lead to a better embedding of the entire graph. <br />


No data is provided for this file. Simply run the file to recreate the results. <br />
