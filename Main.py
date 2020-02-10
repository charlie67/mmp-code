import tsplib95
import networkx as nx
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import AffinityPropagation, Birch
from sklearn.datasets import make_blobs
import numpy as np
from itertools import cycle

# def load_file(file_name) -> nx.Graph:
#     problem: tsplib95.models.Problem = tsplib95.load_problem(file_name)
#     return problem.get_graph()


file_name = "testdata/it16862.tsp"
problem: tsplib95.models.Problem = tsplib95.utils.load_problem(file_name)
X = np.zeros(shape=(problem.dimension, 2))
print(X.shape)
for node in problem.get_nodes():
    print(node, problem.get_display(node))
    X[node - 1, 0] = problem.get_display(node)[0]
    X[node - 1, 1] = problem.get_display(node)[1]

plt.close('all')
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

i = 0
# for node, col in zip(X, colors):
    # print("plot", i)
    # i += 1
    # plt.plot(node[0], node[1], col + '.', markersize=15)

plt.title("All nodes")
plt.show()

# affinity propogation
af = AffinityPropagation().fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)
print('Estimated number of AffinityPropagation clusters: %d' % n_clusters_)
#
# for k, col in zip(range(n_clusters_), colors):
#     class_members = labels == k
#     cluster_center = X[cluster_centers_indices[k]]
#     plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
#     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=14)
#     for x in X[class_members]:
#         plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
#
# plt.title('AffinityPropagation: Estimated number of clusters: %d' % n_clusters_)
plt.show()

# Birch clustering
brc = Birch(branching_factor=50, n_clusters=20, threshold=1.5)
brc.fit(X)
birch_labels = brc.predict(X)
print(birch_labels)
plt.scatter(X[:, 0], X[:, 1], c=birch_labels, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.title('Birch clustering')
plt.show()
