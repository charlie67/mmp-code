from matplotlib import pyplot as plt

from options.options_holder import Options


def plot_nodes(array, file_name, program_options: Options):
    for node in array:
        plt.plot(node[0], node[1], 'b.', markersize=4)
    plt.title(file_name + " All nodes")
    plt.savefig(program_options.OUTPUT_DIRECTORY + file_name + "-all-nodes.png", dpi=program_options.PLT_DPI_VALUE)

    if program_options.DISPLAY_PLOTS:
        plt.show()


def plot_tour_for_cluster(cluster_tour, cluster_nodes, program_options: Options, cluster_title, entry_exit_nodes):
    figure = plt.figure()

    for node in cluster_tour:
        marker_face_color = "r"

        if node in entry_exit_nodes:
            marker_face_color = "g"

        plt.plot(cluster_nodes[node][0], cluster_nodes[node][1], 'o', markerfacecolor=marker_face_color,
                 markeredgecolor='k',
                 markersize=4,
                 figure=figure)

    c = 0
    for i in cluster_tour:
        tour_node_next_value = max((c-1), 0)
        j = cluster_tour[tour_node_next_value]
        plt.plot([cluster_nodes[i][0], cluster_nodes[j][0]], [cluster_nodes[i][1], cluster_nodes[j][1]], 'k',
                 linewidth=0.5)
        c += 1

    plt.title("Tour of " + cluster_title)

    if program_options.DISPLAY_PLOTS:
        plt.show()

    plt.savefig(
        program_options.OUTPUT_DIRECTORY + program_options.TSP_PROBLEM_NAME + "-tour-cluster-" + cluster_title + ".png",
        dpi=program_options.PLT_DPI_VALUE)
    plt.close()


def plot_aco_clustered_tour(tour, clustered_data, program_options: Options):
    nodes_in_tour = clustered_data.get_all_cluster_centres_and_unclassified_node_locations()
    figure = plt.figure()

    for i in range(len(tour)):
        plt.plot(nodes_in_tour[i][0], nodes_in_tour[i][1], 'o', markerfacecolor="r", markeredgecolor='k', markersize=8,
                 figure=figure)
        plt.annotate(i, xy=(nodes_in_tour[i][0], nodes_in_tour[i][1]), fontsize=5, ha='center', va='center')

    c = 0
    for i in tour:
        j = tour[c - 1]
        plt.plot([nodes_in_tour[i][0], nodes_in_tour[j][0]], [nodes_in_tour[i][1], nodes_in_tour[j][1]], 'k',
                 linewidth=0.5)
        c += 1

    plt.title("Tour of clustered nodes")
    if program_options.DISPLAY_PLOTS:
        plt.show(dpi=program_options.PLT_DPI_VALUE)
    plt.savefig(program_options.OUTPUT_DIRECTORY + program_options.TSP_PROBLEM_NAME + "-aco-clustered-tour.png", dpi=program_options.PLT_DPI_VALUE)
    plt.close()


def plot_complete_tsp_tour(tour, node_id_to_coordinate_dict, title, program_options: Options,
                           node_id_shown=False):
    num = 0
    figure = plt.figure()
    for i in tour:
        j = tour[num - 1]

        node_i = node_id_to_coordinate_dict[i]
        node_j = node_id_to_coordinate_dict[j]

        plt.plot(node_i[0], node_i[1], 'b.', markersize=3, figure=figure)
        plt.plot([node_i[0], node_j[0]], [node_i[1], node_j[1]], 'k', linewidth=0.5, figure=figure)
        if node_id_shown:
            plt.annotate(i, xy=(node_i[0], node_i[1]), fontsize=4, ha='center', va='center')
        num += 1
    plt.title(title)
    plt.savefig(program_options.OUTPUT_DIRECTORY + program_options.TSP_PROBLEM_NAME + title + "-solution.png", dpi=program_options.PLT_DPI_VALUE)
    if program_options.DISPLAY_PLOTS:
        plt.show()

    plt.close()
