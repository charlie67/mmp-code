import random
import numpy as np

from matplotlib import pyplot as plt

from distance_calculation import aco_distance_callback

if __name__ == "__main__":
    number_nodes = 1000
    problem_name = str(number_nodes) + "_node"
    output_directory = "testdata/created/"

    x_value_min = 0
    x_value_max = 50000

    y_value_min = 0
    y_value_max = 50000

    thirtieth_total_size_x = (x_value_max - x_value_min) / 30
    thirtieth_total_size_y = (y_value_max - y_value_min) / 30

    # The file name that this should be saved to
    file_name = output_directory + problem_name + ".tsp"
    graph_name = output_directory + problem_name + ".png"

    # Stores all the nodes that have been generated
    nodes = list()

    # There are number_clusters number of clusters and the size of these cluster is between cluster_size_min and
    # cluster_size_max
    number_clusters = 10

    # Setting these higher gives you more dense clusters
    cluster_size_min = 60
    cluster_size_max = 70

    # Stores the sizes of all the clusters
    cluster_sizes = list()
    # The size of the nodes in all the clusters
    total_size_of_all_clusters = 0

    # Calculate the sizes of all the clusters
    for i in range(number_clusters):
        size = random.randint(cluster_size_min, cluster_size_max)
        cluster_sizes.append(size)
        total_size_of_all_clusters += size

    number_nodes_to_generate = number_nodes - total_size_of_all_clusters

    for i in range(number_nodes_to_generate):
        x_location = random.randint(x_value_min, x_value_max)
        y_location = random.randint(y_value_min, y_value_max)

        node = [x_location, y_location]
        nodes.append(node)

    # The location of all the clusters
    location_of_clusters = []

    # The nodes that belong to clusters, these get added to this and then combined when all the clusters are created
    # so that clusters don't get created too close to each other
    nodes_in_clusters = []

    # Need to generate the clusters
    # Pick number_clusters number of random points and generate the clusters there
    for i in range(number_clusters):
        # This is where the cluster will be located
        cluster_location = nodes[random.randint(0, len(nodes) - 1)]

        # Need to check the cluster_location is good and not too near any other clusters
        cluster_location_is_good = False
        while not cluster_location_is_good:
            cluster_location_is_good = True
            for location in location_of_clusters:
                distance = aco_distance_callback(location, cluster_location)
                if distance < thirtieth_total_size_x*10:
                    # too close so need to try and find a new cluster location
                    cluster_location_is_good = False

            cluster_location = nodes[random.randint(0, len(nodes) - 1)]

        location_of_clusters.append(cluster_location)
        size = cluster_sizes[i]

        cluster_x_value_min = cluster_location[0] - thirtieth_total_size_x
        cluster_x_value_max = cluster_location[0] + thirtieth_total_size_x

        cluster_y_value_min = cluster_location[1] - thirtieth_total_size_y
        cluster_y_value_max = cluster_location[1] + thirtieth_total_size_y

        for j in range(size):
            x_location = random.uniform(cluster_x_value_min, cluster_x_value_max)
            y_location = random.uniform(cluster_y_value_min, cluster_y_value_max)

            node = [x_location, y_location]
            nodes_in_clusters.append(node)

    # When cluster creation has been done add the cluster nodes on
    nodes.extend(nodes_in_clusters)

    # Check that the correct number of nodes have been generated
    node_diff = number_nodes - len(nodes)
    assert node_diff == 0

    # Plot the problem
    figure = plt.figure()
    for node in nodes:
        plt.plot(node[0], node[1], 'o', markersize=5, figure=figure)

    plt.savefig(graph_name, dpi=800)
    plt.close(fig=figure)

    # Save it to a tsp file
    with open(file_name, 'w') as file:
        file.write("NAME: " + problem_name + "\n")
        file.write("COMMENT: auto generated TSP problem\n")
        file.write("COMMENT: Contains " + str(number_nodes) + " nodes\n")
        file.write("COMMENT: Seeded with " + str(number_clusters) + " clusters\n")
        file.write("TYPE: TSP\n")
        file.write("DIMENSION: " + str(number_nodes) + "\n")
        file.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        file.write("NODE_COORD_SECTION\n")

        counter = 1
        for node in nodes:
            file.write(str(counter) + " " + str(node[0]) + " " + str(node[1]) + "\n")
            counter += 1
