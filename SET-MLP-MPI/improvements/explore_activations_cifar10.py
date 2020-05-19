import matplotlib
#matplotlib.use('Agg')
from scipy import stats
from scipy.sparse import csr_matrix, find
import matplotlib.pyplot as plt
import datetime
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
from models.set_mlp_sequential import *
from utils.load_data import *

def weighted_graph_from_weight_masks(weight_masks):
    nodes = []
    nodes.append(["0-{}".format(i) for i in range(weight_masks[0].shape[0])])
    graph = nx.Graph()
    graph.add_nodes_from(nodes[0])
    for layer_number, weight_mask in enumerate(weight_masks):
        left, right = weight_mask.shape
        nodes.append(["{}-{}".format(layer_number + 1, i) for i in range(right)])
        graph.add_nodes_from(nodes[layer_number + 1])
        graph.add_weighted_edges_from(
            [(nodes[layer_number][i], nodes[layer_number + 1][j], weight_mask[i][j]) for i in range(left) for j in
             range(right)])
    return graph


def unweighted_graph_from_weight_masks(weight_masks):
    nodes = []
    nodes.append(["0-{}".format(i) for i in range(weight_masks[0].shape[0])])
    graph = nx.Graph()
    graph.add_nodes_from(nodes[0])
    for layer_number, weight_mask in enumerate(weight_masks):
        left, right = weight_mask.shape
        nodes.append(["{}-{}".format(layer_number + 1, i) for i in range(right)])
        graph.add_nodes_from(nodes[layer_number + 1])
        graph.add_edges_from(
            [(nodes[layer_number][i], nodes[layer_number + 1][j]) for i in range(left) for j in range(right) if
             weight_mask[i][j] != 0])
    return graph

X_train, Y_train, X_test, Y_test = load_cifar10_data(50000, 10000)
connections = np.load("../Results/set_mlp_sequential_softmax_mnist_60000_training_samples_e20_rand0_input_connections.npz")["inputLayerConnections"]
weights = np.load("../Results/set_mlp_sequential_cifar10_50000_training_samples_e20_rand0_weights.npz",  allow_pickle=True)
biases = np.load("../Results/set_mlp_sequential_cifar10_50000_training_samples_e20_rand0_biases.npz", allow_pickle=True)
w10 = weights['arr_9'].item()
b10 = biases['arr_9'].item()

weight_masks = np.array([w10[1].toarray(), w10[2].toarray(), w10[3].toarray(), w10[4].toarray()])
#print("Started creating graph at", datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))
#graph = weighted_graph_from_weight_masks(weight_masks)
#print("Graph created, start calculating centrality",
#                       datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))
#centrality = nx.edge_current_flow_betweenness_centrality(graph)
#print("Centrality calculated", datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))
#plt.figure()
#np.savetxt("Centrality_values.txt", centrality.values())

# nx.draw(graph)
# plt.savefig("filename.png")
# plt.show()
config = {
            'n_processes': 3,
            'n_epochs': 10,
            'batch_size': 100,
            'dropout_rate': 0.3,
            'seed': 0,
            'lr': 0.01,
            'lr_decay': 0.0,
            'zeta': 0.3,
            'epsilon': 20,
            'momentum': 0.9,
            'weight_decay': 0.0,
            'n_hidden_neurons': 1000,
            'n_training_samples': 60000,
            'n_testing_samples': 10000,
            'loss': 'cross_entropy'
        }
set_mlp = SET_MLP((X_train.shape[1], 4000, 1000, 4000,
                           Y_train.shape[1]), (Relu, Relu, Relu, Softmax), **config)

set_mlp.w = w10
set_mlp.b = b10

print("\nNon zero before pruning: ")
for k, w in w10.items():
    print(w.count_nonzero())

accuracy, activations_test = set_mlp.predict(X_test, Y_test, batch_size=1)
print("\nAccuracy before pruning on the testing data: ", accuracy)
loss_test = set_mlp.loss.loss(Y_test, activations_test)
print(f"Loss test: {loss_test}")

for k, w in w10.items():
    i, j, v = find(w)
    # plt.hist(np.round(v,2), bins=100)
    # plt.title(f'Weight distribution layer {k}')
    # plt.xlabel("value")
    # plt.ylabel("Frequency")
    # plt.show()
    weights = w.toarray()
    positive_mean = np.median(weights[weights > 0])
    # positive_std = np.std(weights[weights > 0])
    # zscore_pos = stats.zscore(v)
    #
    # plt.plot(zscore_pos)
    # plt.show()

    negative_mean = np.median(weights[weights < 0])

    p95 = np.percentile(v, 95)
    p75 = np.percentile(v, 75)
    p50 = np.percentile(v, 50)
    p25 = np.percentile(v, 25)
    p5 = np.percentile(v, 5)

    p20 = np.percentile(v, 20)
    p80 = np.percentile(v, 80)

    # negative_std = np.std(weights[weights < 0])
    # zscore_neg = stats.zscore(weights[weights < 0])
    eps = 0.08
    #weights[((weights > positive_mean - eps) & (weights < positive_mean + eps)) & (weights > 0)] = 0.0
    # weights[((weights > negative_mean - eps) & (weights < negative_mean + eps)) & (weights < 0)] = 0.0
    weights[(weights <= np.round(p75,  2)) & (weights > 0)] = 0.0
    weights[(weights >= np.round(p25,  2)) & (weights < 0)] = 0.0
    #weights[np.abs(weights) <= eps] = 0.0
    # weights[(np.abs(np.round(weights, 2)) == np.round(positive_mean, 2)) | (np.abs(np.round(weights, 2)) == np.round(negative_mean, 2))] = 0.0
    # weights[(np.round(weights, 2) != np.round(p5, 2)) & (np.round(weights, 2) != np.round(p25, 2)) &
    #         (np.round(weights, 2) != np.round(p50, 2)) & (np.round(weights, 2) != np.round(p75, 2)) & (np.round(weights, 2) != np.round(p95, 2))] = 0.0
    #weights[np.round(weights, 2) == np.round(p5,  2)] = 0.0
    weights[np.round(weights, 2) == np.round(p25, 2)] = 0.0
    # weights[np.round(weights, 2) == np.round(p50, 2)] = 0.0
    weights[np.round(weights, 2) == np.round(p75, 2)] = 0.0
    #weights[np.round(weights, 2) == np.round(p95, 2)] = 0.0
    w = csr_matrix(weights)
    i, j, v = find(w)
    # plt.hist(v, bins=100)
    # plt.title(f'Weight distribution layer {k}')
    # plt.xlabel("value")
    # plt.ylabel("Frequency")
    # plt.show()
    w10[k] = w


set_mlp.w = w10
set_mlp.b = b10

print("\nNon zero after pruning: ")
for k, w in w10.items():
    print(w.count_nonzero())

accuracy, activations_test = set_mlp.predict(X_test, Y_test, batch_size=1)
print("\nAccuracy after pruning on the testing data: ", accuracy)
loss_test = set_mlp.loss.loss(Y_test, activations_test)
print(f"Loss test: {loss_test}")
