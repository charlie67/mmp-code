import os

import tsplib95
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation, Birch, KMeans, DBSCAN
import numpy as np
from itertools import cycle

from ClusteredData import ClusteredData, Cluster

NUMBER_CLUSTERS = 5


def plot_nodes(array, plot_colors, file_name):
    for node, col in zip(array, plot_colors):
        plt.plot(node[0], node[1], 'b.', markersize=10)
    plt.title(file_name + " All nodes")
    plt.savefig(file_name + "-nodes.png")
    plt.show()


def perform_affinity_propagation(data) -> ClusteredData:
    # The data that will be returned
    clustered_data = ClusteredData(data, list())

    af = AffinityPropagation(convergence_iter=500, max_iter=20000).fit(data)
    affinity_propagation_cluster_centers_indices = af.cluster_centers_indices_
    affinity_propagation_labels = af.labels_
    n_clusters_ = len(affinity_propagation_cluster_centers_indices)
    print('Estimated number of AffinityPropagation clusters: %d' % n_clusters_)

    for k in range(n_clusters_):
        class_members = affinity_propagation_labels == k
        cluster_center = data[affinity_propagation_cluster_centers_indices[k]]

        cluster = Cluster(cluster_centre=cluster_center, nodes=data[class_members])
        clustered_data.add_cluster(cluster)

    return clustered_data


def perform_k_means_clustering(data) -> ClusteredData:
    # The data that will be returned
    clustered_data = ClusteredData(data, list())

    km = KMeans(init='k-means++', n_clusters=NUMBER_CLUSTERS, n_init=10)
    km.fit(data)
    k_mean_labels = km.predict(data)
    k_means_cluster_centers_indices = km.cluster_centers_
    n_clusters_ = len(k_means_cluster_centers_indices)
    for k in range(n_clusters_):
        class_members = k_mean_labels == k
        cluster = Cluster(cluster_centre=k_means_cluster_centers_indices[k], nodes=data[class_members])
        clustered_data.add_cluster(cluster)

    print("k-mean clusters", k_mean_labels)
    return clustered_data


def perform_birch_clustering(data) -> ClusteredData:
    # The data that will be returned
    clustered_data = ClusteredData(data, list())

    brc = Birch(branching_factor=50, n_clusters=NUMBER_CLUSTERS, threshold=0.5)
    brc.fit(data)
    birch_labels = brc.predict(data)
    birch_cluster_centers_indices = brc.subcluster_centers_

    n_clusters_ = len(birch_cluster_centers_indices)
    for k in range(n_clusters_):
        class_members = birch_labels == k
        cluster = Cluster(cluster_centre=birch_cluster_centers_indices[k], nodes=data[class_members])
        clustered_data.add_cluster(cluster)

    print("birch clusters", birch_labels)
    plt.scatter(data[:, 0], data[:, 1], c=birch_labels, cmap='rainbow', alpha=0.7,
                edgecolors=None)
    plt.title(file_name + ' Birch clustering')
    plt.savefig(file_name + "-birch-clustering.png")
    plt.show()

    return clustered_data


def plot_clustered_graph(file_name, plot_colours, cluster: ClusteredData, cluster_type):
    # This plotting was adapted from the affinity propagation sklearn example
    for k, col in zip(range(len(cluster.get_clusters())), plot_colours):
        class_members = cluster.get_clusters()[k].get_nodes()
        cluster_center = cluster.get_clusters()[k].get_cluster_centre()

        plt.plot(class_members[:, 0], class_members[:, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)

        for x in class_members:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col, linewidth=0.5)

    plt.title(file_name + ' ' + cluster_type + ': clusters: %d' % len(cluster.get_clusters()))
    plt.savefig(file_name + "-" + cluster_type + "-clustering.png")
    plt.show()


def move_between_two_clusters(cluster_a: Cluster, cluster_b: Cluster):
    # move from cluster_a to cluster_b
    # get the centroid of cluster b
    # and then find the node in cluster_a that is closest to this centroid
    # then find the node in cluster_b that is closest to this new closest cluster_a value

    closest_cluster_a = None
    closest_cluster_a_distance = None

    centre = cluster_b.get_cluster_centre()
    for node in cluster_a.get_nodes():
        # calculate the distance between node and centre
        distance = np.linalg.norm(node - centre)

        if closest_cluster_a_distance is None or distance < closest_cluster_a_distance or closest_cluster_a is None:
            closest_cluster_a_distance = distance
            closest_cluster_a = node

    closest_cluster_b = None
    closest_cluster_b_distance = None

    for node in cluster_b.get_nodes():
        # calculate the distance between node and centre
        distance = np.linalg.norm(node - closest_cluster_a)

        if closest_cluster_b_distance is None or distance < closest_cluster_b_distance or closest_cluster_b is None:
            closest_cluster_b_distance = distance
            closest_cluster_b = node

    plt.plot(closest_cluster_a[0], closest_cluster_a[1], 'o', markerfacecolor="r",
             markeredgecolor='k', markersize=14)

    plt.plot(closest_cluster_b[0], closest_cluster_b[1], 'o', markerfacecolor="b",
             markeredgecolor='k', markersize=14)
    plt.title("test movement between cluster 1 and 2")
    plt.show()


if __name__ == '__main__':
    directory_name = "testdata/world/"

    for file in os.listdir(directory_name):
        if not file.endswith(".tsp"):
            continue
        file_name = directory_name + file
        problem: tsplib95.models.Problem = tsplib95.utils.load_problem(file_name)
        problem_data_array = np.zeros(shape=(problem.dimension, 2))

        # transform the problem data into a numpy array for scikit learn to use
        for node in problem.get_nodes():
            problem_data_array[node - 1, 0] = problem.get_display(node)[0]
            problem_data_array[node - 1, 1] = problem.get_display(node)[1]

        plt.close('all')
        plt.figure(1)
        plt.clf()

        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

        plot_nodes(problem_data_array, colors, file_name)

        # affinity propagation
        affinity_propagation_clustered_data = perform_affinity_propagation(problem_data_array)
        plot_clustered_graph(file_name, colors, cluster=affinity_propagation_clustered_data, cluster_type="Affinity-Propagation")

        move_between_two_clusters(affinity_propagation_clustered_data.get_clusters()[0], affinity_propagation_clustered_data.get_clusters()[1])

        # K-means clustering
        k_means_clustered_data = perform_k_means_clustering(problem_data_array)
        plot_clustered_graph(file_name, colors, cluster=k_means_clustered_data, cluster_type="K-Means")

        # Birch clustering
        birch_clustered_data = perform_birch_clustering(problem_data_array)
        plot_clustered_graph(file_name, colors, cluster=birch_clustered_data, cluster_type="Birch")
