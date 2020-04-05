from matplotlib import pyplot as plt

from clustering.clustered_data import ClusteredData


def plot_nodes(array, file_name, output_location, display_plot=True):
    for node in array:
        plt.plot(node[0], node[1], 'b.', markersize=10)
    plt.title(file_name + " All nodes")
    plt.savefig(output_location + file_name + "-all-nodes.png")

    if display_plot:
        plt.show()


def plot_aco_clustered_tour(tour, clustered_data: ClusteredData, tsp_problem_name, output_directory, display_plot=True):
    nodes_in_tour = clustered_data.get_all_cluster_centres_and_unclassified_node_locations()
    figure = plt.figure(figsize=[40, 40])

    for i in range(len(tour)):
        plt.plot(nodes_in_tour[i][0], nodes_in_tour[i][1], 'o', markerfacecolor="r", markeredgecolor='k', markersize=14,
                 figure=figure)
        plt.annotate(i, xy=(nodes_in_tour[i][0], nodes_in_tour[i][1]), fontsize=10, ha='center', va='center')

    c = 0
    for i in tour:
        j = tour[c - 1]
        plt.plot([nodes_in_tour[i][0], nodes_in_tour[j][0]], [nodes_in_tour[i][1], nodes_in_tour[j][1]], 'k',
                 linewidth=0.5)
        c += 1

    plt.title("Tour of clustered nodes")
    if display_plot:
        plt.show()
    plt.savefig(output_directory + tsp_problem_name + "-aco-clustered-tour.png")
    plt.close()


def plot_complete_tsp_tour(tour, node_id_to_coordinate_dict, title, tsp_problem_name, output_directory,
                           node_id_shown=False, display_plot=True):
    num = 0
    figure = plt.figure(figsize=[40, 40])
    for i in tour:
        j = tour[num - 1]

        node_i = node_id_to_coordinate_dict[i]
        node_j = node_id_to_coordinate_dict[j]

        plt.plot(node_i[0], node_i[1], 'b.', markersize=10, figure=figure)
        plt.plot([node_i[0], node_j[0]], [node_i[1], node_j[1]], 'k', linewidth=0.5, figure=figure)
        if node_id_shown:
            plt.annotate(i, xy=(node_i[0], node_i[1]), fontsize=10, ha='center', va='center')
        num += 1
    plt.title(title)
    plt.savefig(output_directory + tsp_problem_name + title + "-solution.png")
    if display_plot:
        plt.show()

    plt.close()
