perform_aco_over_clustered_problem(clustered_data.turn_clusters_into_nx_graph(problem))


# distance matrix of all the nodes
adjacency_matrix = nx.to_numpy_matrix(problem_graph)

average_value_counter = 0
largest_distance = adjacency_matrix.max()

# make the smallest number the max to start with so initially every value is smaller
smallest_distance = largest_distance
for x in np.nditer(adjacency_matrix):
    if x == 0:
        continue
    average_value_counter += x
    if x < smallest_distance and x != 0:
        smallest_distance = x

average_distance_value = average_value_counter / adjacency_matrix.size

automated_eps_value = largest_distance/average_distance_value
# program_options.DBSCAN_EPS = automated_eps_value
print("eps could now be " + str(automated_eps_value))