import os
from datetime import datetime
import logging

from itertools import cycle

from clustering.clustering import plot_clustered_graph, perform_affinity_propagation, perform_k_means_clustering, \
    perform_birch_clustering, perform_dbscan_clustering, perform_optics_clustering
from clustering.clustering_algorithm_type_enum import ClusterAlgorithmType
from distance_calculation import calculate_distance_for_tour, aco_distance_callback

from graph_plotting import plot_nodes, plot_aco_clustered_tour, plot_complete_tsp_tour

from loading import load_problem_into_np_array

from tsp_2_opt.tsp_2_opt_improver import run_2_opt
from tsp_2_opt.tsp_2_opt_graph_plotter import TSP2OptAnimator

from aco.multithreaded.multi_threaded_ant_colony import AntColony

if __name__ == '__main__':
    cluster_type_to_use = ClusterAlgorithmType.DBSCAN
    should_display_plots = False

    tsp_problem_name = "dj38.tsp"
    file_name = "testdata/world/" + tsp_problem_name
    problem, problem_data_array = load_problem_into_np_array(file_name)

    # Create the ouput directory where all the graphs are saved
    output_directory = "output/" + tsp_problem_name + str(cluster_type_to_use.value) + str(
        datetime.date(datetime.now())) + str(
        datetime.time(datetime.now())) + "/"
    directory = os.path.dirname(output_directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # setup the logging to a file
    logging.basicConfig(filename=output_directory + 'log.log', level=logging.DEBUG)
    # Remove all the plt debug messages
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    # Create the output directory where the 2-opt animation is saved
    output_directory_2_opt_animation = output_directory + "2-opt-animation/"
    directory = os.path.dirname(output_directory_2_opt_animation)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # key is the node location and the value is the node id
    node_location_to_id_dict = dict()

    # Key is the node id and the value is the node location
    node_id_to_location_dict = dict()

    counter = 0

    for node in problem_data_array:
        node_location_to_id_dict[repr(node)] = counter
        node_id_to_location_dict[counter] = node
        counter += 1

    colors = cycle('bgrcmybgrcmybgrcmybgrcmy')

    plot_nodes(problem_data_array, tsp_problem_name, output_directory, display_plot=should_display_plots)

    clustered_data = None

    if cluster_type_to_use is ClusterAlgorithmType.K_MEANS:
        clustered_data = perform_k_means_clustering(problem_data_array)

    if cluster_type_to_use is ClusterAlgorithmType.AFFINITY_PROPAGATION:
        clustered_data = perform_affinity_propagation(problem_data_array)

    if cluster_type_to_use is ClusterAlgorithmType.BIRCH:
        clustered_data = perform_birch_clustering(problem_data_array)

    if cluster_type_to_use is ClusterAlgorithmType.DBSCAN:
        clustered_data = perform_dbscan_clustering(problem_data_array)

    if cluster_type_to_use is ClusterAlgorithmType.OPTICS:
        clustered_data = perform_optics_clustering(problem_data_array)

    plot_clustered_graph(tsp_problem_name, output_directory, colors, cluster_data=clustered_data,
                         cluster_type=str(cluster_type_to_use), display_plot=should_display_plots)

    # Set the overall node dicts onto the clustering object
    clustered_data.node_location_to_id_dict = node_location_to_id_dict
    clustered_data.node_id_to_location_dict = node_id_to_location_dict

    cluster_nodes = clustered_data.get_dict_node_id_location_mapping_aco()
    cluster_nodes_dict = clustered_data.get_dict_node_id_location_mapping_aco()

    before = datetime.now()
    colony = AntColony(nodes=cluster_nodes_dict, distance_callback=aco_distance_callback, alpha=1, beta=10)
    answer = colony.mainloop()
    after = datetime.now()

    dif = after - before
    logging.debug("Time taken for multithreaded aco %s", dif)

    plot_aco_clustered_tour(answer, clustered_data, display_plot=should_display_plots)
    clustered_data.aco_cluster_tour = answer

    clustered_data.find_nodes_to_move_between_clusters()

    clustered_data.find_tours_within_clusters()
    tour_node_coordinates = clustered_data.get_ordered_nodes_for_all_clusters()

    logging.debug("final tour is %s", tour_node_coordinates)

    counter = 0
    tour_node_id = []

    for node in tour_node_coordinates:
        tour_node_id.append(node_location_to_id_dict[repr(node)])

    clustered_data.node_level_tour = tour_node_id

    tour_node_id_set = set(tour_node_id)
    valid = len(tour_node_id) == len(tour_node_id_set)
    logging.debug("Tour tour as node id %s", tour_node_id)
    logging.debug("Tour is valid %s", valid)

    length_before = calculate_distance_for_tour(tour_node_id, node_id_to_location_dict)
    logging.debug("Length before 2-opt is %s", length_before)

    plot_complete_tsp_tour(tour_node_id, node_id_to_location_dict,
                           title="TSP Tour Before 2-opt. Length: " + str(length_before),
                           tsp_problem_name=tsp_problem_name, output_directory=output_directory,
                           display_plot=should_display_plots)

    tsp_2_opt_graph_animator = TSP2OptAnimator(node_id_to_location_dict)

    before = datetime.now()
    final_route = run_2_opt(existing_route=tour_node_id, node_id_to_location_dict=node_id_to_location_dict,
                            distance_calculator_callback=calculate_distance_for_tour,
                            tsp_2_opt_animator=tsp_2_opt_graph_animator)
    after = datetime.now()

    dif = after - before
    logging.debug("Time taken for 2-opt %s", dif)

    length_after = calculate_distance_for_tour(final_route, node_id_to_location_dict)
    logging.debug("Length after 2-opt is %s", length_after)
    logging.debug("All 2-opt tours %s", tsp_2_opt_graph_animator.tour_history)

    logging.debug("Final route after 2-opt is %s", final_route)
    plot_complete_tsp_tour(final_route, node_id_to_location_dict,
                           title="TSP Tour After 2-opt. Length: " + str(length_after),
                           tsp_problem_name=tsp_problem_name, output_directory=output_directory,
                           display_plot=should_display_plots)

    plot_complete_tsp_tour(final_route, node_id_to_location_dict, title="Final TSP Tour With Node ID",
                           node_id_shown=True, tsp_problem_name=tsp_problem_name, output_directory=output_directory,
                           display_plot=should_display_plots)

    # Create an animation of the 2-opt incremental improvement
    tsp_2_opt_graph_animator.animate(tsp_problem_name, output_directory, output_directory_2_opt_animation)
