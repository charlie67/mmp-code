import os

import networkx as nx
import tsplib95
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
import acopy

from ClusteredData import ClusteredData, Cluster
from Clustering import plot_clustered_graph, perform_affinity_propagation, perform_optics_clustering, \
    perform_k_means_clustering, perform_birch_clustering, perform_dbscan_clustering


def plot_nodes(array, file_name):
    for node in array:
        plt.plot(node[0], node[1], 'b.', markersize=10)
    plt.title(file_name + " All nodes")
    plt.savefig(file_name + "-nodes.png")
    plt.show()


# Find the shortest path between two given clusters. Will find the node in cluster a that is closest to the centre of b
# and then find the node in b that is closest to this node in a
# Will return the numpy coordinates of the two closest clusters
def move_between_two_clusters(cluster_a: Cluster, cluster_b: Cluster):
    # move from cluster_a to cluster_b
    # get the centroid of cluster b
    # and then find the node in cluster_a that is closest to this centroid
    # then find the node in cluster_b that is closest to this new closest cluster_a value

    closest_cluster_a = None
    closest_cluster_a_distance = None
    cluster_a_node_number = None

    centre = cluster_b.get_cluster_centre()
    counter = 0
    for node in cluster_a.get_nodes():
        # calculate the distance between node and centre
        distance = np.linalg.norm(node - centre)

        if (closest_cluster_a_distance is None or distance < closest_cluster_a_distance or closest_cluster_a is None) and (counter not in cluster_a.entry_exit_nodes or len(cluster_a.nodes) == 1):
            closest_cluster_a_distance = distance
            closest_cluster_a = node
            cluster_a_node_number = counter
        counter += 1

    closest_cluster_b = None
    closest_cluster_b_distance = None
    cluster_b_node_number = None

    counter = 0
    for node in cluster_b.get_nodes():
        # calculate the distance between node and centre
        distance = np.linalg.norm(node - closest_cluster_a)

        if (closest_cluster_b_distance is None or distance < closest_cluster_b_distance or closest_cluster_b is None) and (counter not in cluster_b.entry_exit_nodes or len(cluster_b.nodes) == 1):
            closest_cluster_b_distance = distance
            closest_cluster_b = node
            cluster_b_node_number = counter
        counter += 1

    cluster_a.entry_exit_nodes.append(cluster_a_node_number)
    cluster_b.entry_exit_nodes.append(cluster_b_node_number)

    return closest_cluster_a, closest_cluster_b


# Go over the ACO tour and find a path connecting each cluster
def find_movement_between_clusters(tour, clustered_data: ClusteredData, all_node_array):
    for node in all_node_array:
        plt.plot(node[0], node[1], 'b.', markersize=10)

    c = 0
    nodes_in_tour = clustered_data.get_all_overall_nodes_as_clusters()

    for node in tour:
        j = tour[c - 1]
        closest_a, closest_b = move_between_two_clusters(nodes_in_tour[node], nodes_in_tour[j])

        plt.plot([closest_a[0], closest_b[0]], [closest_a[1], closest_b[1]], 'k', linewidth=0.5)

        c += 1
    plt.title("ACO ")
    plt.show()


def plot_tour(tour, clustered_data: ClusteredData):
    nodes_in_tour = clustered_data.get_all_overall_nodes()
    for i in range(len(tour)):
        plt.plot(nodes_in_tour[i][0], nodes_in_tour[i][1], 'o', markerfacecolor="r", markeredgecolor='k', markersize=14)
        plt.annotate(i, xy=(nodes_in_tour[i][0], nodes_in_tour[i][1]), fontsize=10, ha='center', va='center')

    c = 0
    for i in tour:
        j = tour[c - 1]
        plt.plot([nodes_in_tour[i][0], nodes_in_tour[j][0]], [nodes_in_tour[i][1], nodes_in_tour[j][1]], 'k',
                 linewidth=0.5)
        c += 1

    plt.title("Tour of clustered nodes")
    plt.show()


def calculate_shortest_paths_for_cluster_using_aco(cluster: Cluster):
    solver = acopy.Solver(rho=.03, q=1)
    colony = acopy.Colony(alpha=1, beta=10)

    # printout_plugin = acopy.plugins.Printout()
    # solver.add_plugin(printout_plugin)

    graph = cluster.turn_cluster_into_networkx_graph()
    cluster_tour = solver.solve(graph, colony, limit=300)

    print(cluster_tour)


def perform_aco_over_clustered_problem():
    solver = acopy.Solver(rho=.03, q=1)
    colony = acopy.Colony(alpha=1, beta=10)
    printout_plugin = acopy.plugins.Printout()
    solver.add_plugin(printout_plugin)
    return solver.solve(graph, colony, limit=300)


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
        # affinity_propagation_clustered_data = perform_affinity_propagation(problem_data_array)
        # plot_clustered_graph(file_name, colors, cluster_data=affinity_propagation_clustered_data,
        #                      cluster_type="Affinity-Propagation")

        # K-means clustering
        # k_means_clustered_data = perform_k_means_clustering(problem_data_array)
        # plot_clustered_graph(file_name, colors, cluster_data=k_means_clustered_data, cluster_type="K-Means")

        # Birch clustering
        # birch_clustered_data = perform_birch_clustering(problem_data_array)
        # plot_clustered_graph(file_name, colors, cluster_data=birch_clustered_data, cluster_type="Birch")

        # DBSCAN clustering
        # dbscan_clustered_data = perform_dbscan_clustering(problem_data_array)
        # plot_clustered_graph(file_name, colors, cluster_data=dbscan_clustered_data, cluster_type="DBSCAN")

        # OPTICS clustering
        optics_clustered_data = perform_optics_clustering(problem_data_array)
        plot_clustered_graph(file_name, colors, cluster_data=optics_clustered_data, cluster_type="OPTICS")

        graph = optics_clustered_data.turn_clusters_into_nx_graph(tsplib_problem=problem)

        tour = perform_aco_over_clustered_problem()
        plot_tour(tour.nodes, optics_clustered_data)

        find_movement_between_clusters(tour.nodes, optics_clustered_data, problem_data_array)

        calculate_shortest_paths_for_cluster_using_aco(cluster=optics_clustered_data.clusters[0])

        print(tour.nodes)
