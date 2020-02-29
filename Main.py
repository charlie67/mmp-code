import os

import tsplib95
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation, Birch, KMeans, DBSCAN
import numpy as np
from itertools import cycle
import acopy

from ClusteredData import ClusteredData, Cluster

NUMBER_CLUSTERS = 20


def plot_nodes(array, file_name):
    for node in array:
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

    for k in range(brc.n_clusters):
        class_members = birch_labels == k
        nodes_in_cluster = data[class_members]
        # birch has no way of telling you the final cluster centres so have to calculate it yourself
        cluster_centre = nodes_in_cluster.mean(axis=0)
        cluster = Cluster(cluster_centre=cluster_centre, nodes=nodes_in_cluster)
        clustered_data.add_cluster(cluster)

    print("birch clusters", birch_labels)

    return clustered_data


def perform_dbscan_clustering(data) -> ClusteredData:
    # The data that will be returned
    clustered_data = ClusteredData(data, list())

    db = DBSCAN(eps=50, min_samples=3).fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    db_labels = db.labels_
    db_n_clusters_ = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise_ = list(db_labels).count(-1)

    for k in range(db_n_clusters_):
        class_members = db_labels == k
        nodes_in_cluster = data[class_members]
        cluster_centre = nodes_in_cluster.mean(axis=0)
        cluster = Cluster(cluster_centre=cluster_centre, nodes=nodes_in_cluster)
        clustered_data.add_cluster(cluster)

    if n_noise_ > 0:
        class_members = db_labels == -1
        unclassified_nodes = data[class_members]
        clustered_data.set_unclassified_nodes(unclassified_nodes)

    return clustered_data


def plot_clustered_graph(file_name, plot_colours, cluster_data: ClusteredData, cluster_type):
    # This plotting was adapted from the affinity propagation sklearn example
    i = 0
    for k, col in zip(range(len(cluster_data.get_clusters())), plot_colours):
        class_members = cluster_data.get_clusters()[k].get_nodes()
        cluster_center = cluster_data.get_clusters()[k].get_cluster_centre()

        plt.plot(class_members[:, 0], class_members[:, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
        plt.annotate(i, xy=(cluster_center[0], cluster_center[1]), fontsize=10, ha='center', va='center')
        i += 1

        for x in class_members:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col, linewidth=0.5)

    unclassified_nodes = cluster_data.get_unclassified_nodes()
    if len(unclassified_nodes) > 0:
        for k in unclassified_nodes:
            plt.plot(k[0], k[1], 'o', markerfacecolor='k', markeredgecolor='k', markersize=6)

    plt.title(file_name + ' ' + cluster_type + ': clusters: %d' % len(cluster_data.get_clusters()))
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


def plot_tour(tour, clustered_data: ClusteredData):
    nodes_in_tour = clustered_data.get_all_overall_nodes()
    for i in range(len(tour)):
        plt.plot(nodes_in_tour[i][0], nodes_in_tour[i][1], 'o', markerfacecolor="r", markeredgecolor='k', markersize=14)
        plt.annotate(i, xy=(nodes_in_tour[i][0], nodes_in_tour[i][1]), fontsize=10, ha='center', va='center')
        if i > 0:
            j = tour[i-1]
            plt.plot([nodes_in_tour[i][0], nodes_in_tour[j][0]], [nodes_in_tour[i][1], nodes_in_tour[j][1]], 'k', linewidth=0.5)

    plt.title("Overall tour")
    plt.show()


if __name__ == '__main__':
    directory_name = "testdata/world/"

    for file in os.listdir(directory_name):
        if not file.endswith("38.tsp"):
            continue
        file_name = directory_name + file
        problem: tsplib95.models.Problem = tsplib95.utils.load_problem(file_name)
        problem_data_array = np.zeros(shape=(problem.dimension, 2))

        # transform the problem data into a numpy array of node coordinates for scikit learn to use
        for node in problem.get_nodes():
            problem_data_array[node - 1, 0] = problem.get_display(node)[0]
            problem_data_array[node - 1, 1] = problem.get_display(node)[1]

        colors = cycle('bgrcmybgrcmybgrcmybgrcmy')

        plot_nodes(problem_data_array, file_name)

        # affinity propagation
        affinity_propagation_clustered_data = perform_affinity_propagation(problem_data_array)
        plot_clustered_graph(file_name, colors, cluster_data=affinity_propagation_clustered_data, cluster_type="Affinity-Propagation")

        # K-means clustering
        # k_means_clustered_data = perform_k_means_clustering(problem_data_array)
        # plot_clustered_graph(file_name, colors, cluster_data=k_means_clustered_data, cluster_type="K-Means")
        #
        # # Birch clustering
        # birch_clustered_data = perform_birch_clustering(problem_data_array)
        # plot_clustered_graph(file_name, colors, cluster_data=birch_clustered_data, cluster_type="Birch")
        #
        # # DBSCAN clustering
        # dbscan_clustered_data = perform_dbscan_clustering(problem_data_array)
        # plot_clustered_graph(file_name, colors, cluster_data=dbscan_clustered_data, cluster_type="DBSCAN")

        graph = affinity_propagation_clustered_data.turn_clusters_into_nx_graph(tsplib_problem=problem)
        #
        # nx.draw(graph, with_labels=True, font_weight='bold')
        # plt.show()
        #
        solver = acopy.Solver(rho=.03, q=1)
        colony = acopy.Colony(alpha=1, beta=10)

        printout_plugin = acopy.plugins.Printout()
        solver.add_plugin(printout_plugin)

        tour = solver.solve(graph, colony, limit=1000)
        plot_tour(tour.nodes, affinity_propagation_clustered_data)

        print(tour.nodes)
