import os

import matplotlib.pyplot as plt
from itertools import cycle
import acopy

from ClusteredData import ClusteredData, move_between_two_clusters
from Clustering import plot_clustered_graph, perform_affinity_propagation, perform_optics_clustering, \
    perform_k_means_clustering, perform_birch_clustering, perform_dbscan_clustering
from Loading import load_problem_into_np_array


def plot_nodes(array, file_name):
    for node in array:
        plt.plot(node[0], node[1], 'b.', markersize=10)
    plt.title(file_name + " All nodes")
    plt.savefig(file_name + "-nodes.png")
    plt.show()


def plot_tour(tour, clustered_data: ClusteredData):
    nodes_in_tour = clustered_data.get_all_cluster_centres_and_unclassified_node_locations()
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


def perform_aco_over_clustered_problem():
    solver = acopy.Solver(rho=.03, q=1)
    colony = acopy.Colony(alpha=1, beta=10)
    printout_plugin = acopy.plugins.Printout()
    solver.add_plugin(printout_plugin)
    return solver.solve(graph, colony, limit=250)


if __name__ == '__main__':
    directory_name = "testdata/world/"

    for file in os.listdir(directory_name):
        if not file.endswith("929.tsp"):
            continue
        file_name = directory_name + file
        problem, problem_data_array = load_problem_into_np_array(file_name)

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
        dbscan_clustered_data = perform_dbscan_clustering(problem_data_array)
        plot_clustered_graph(file_name, colors, cluster_data=dbscan_clustered_data, cluster_type="DBSCAN")

        # OPTICS clustering
        # optics_clustered_data = perform_optics_clustering(problem_data_array)
        # plot_clustered_graph(file_name, colors, cluster_data=optics_clustered_data, cluster_type="OPTICS")

        clustered_data = dbscan_clustered_data

        graph = clustered_data.turn_clusters_into_nx_graph(tsplib_problem=problem)

        tour = perform_aco_over_clustered_problem()
        tour_nodes = tour.nodes
        print("Tour is", tour, tour_nodes)

        plot_tour(tour.nodes, clustered_data)
        clustered_data.tour = tour_nodes

        clustered_data.find_nodes_to_move_between_clusters()

        clustered_data.find_tours_within_clusters()
        ordered_nodes = clustered_data.get_ordered_nodes_for_all_clusters()
        print("final tour is ", ordered_nodes)

        num = 0

        figure = plt.figure(figsize=[40, 40])
        for i in ordered_nodes:
            j = ordered_nodes[num - 1]
            plt.plot(i[0], i[1], 'b.', markersize=10, figure=figure)
            plt.plot([i[0], j[0]], [i[1], j[1]], 'k', linewidth=0.5, figure=figure)
            num += 1

        plt.title("Complete tour")
        plt.savefig(file_name + "solution.png")
        plt.show()
