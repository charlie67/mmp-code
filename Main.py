import logging
import os
import sys
from datetime import datetime
from itertools import cycle

import acopy
import networkx as nx
import numpy as np

from aco.aco_type_enum import ACOType
from aco.acopy.iteration_plotter_plugin import IterationPlotterPlugin
from aco.acopy.logger_plugin import LoggerPlugin
from aco.multithreaded.multi_threaded_ant_colony import AntColony
from clustering.clustering import plot_clustered_graph, perform_affinity_propagation, perform_k_means_clustering, \
    perform_birch_clustering, perform_dbscan_clustering, perform_optics_clustering
from clustering.clustering_algorithm_type_enum import ClusterAlgorithmType
from distance_calculation import calculate_distance_for_tour, aco_distance_callback
from loading import load_problem_into_np_array
from options.options_holder import Options
from plotting.graph_plotting import plot_nodes, plot_aco_clustered_tour, plot_complete_tsp_tour
from plotting.tour_improvement_plotter import TourImprovementAnimator
from tsp_2_opt.tsp_2_opt_improver import run_2_opt

if __name__ == '__main__':
    program_start_time = datetime.now()

    program_options = Options(cluster_type=ClusterAlgorithmType.DBSCAN, dbscan_eps=50, display_plots=False,
                              aco_type=ACOType.ACO_PY, aco_iterations=100)

    tsp_problem_name = "dj38.tsp"
    file_name = "testdata/world/" + tsp_problem_name
    problem, problem_data_array = load_problem_into_np_array(file_name)
    problem_graph = problem.get_graph()

    # Create the ouput directory where all the graphs are saved
    output_directory = "output/" + tsp_problem_name + str(program_options.CLUSTER_TYPE.value) + str(
        datetime.date(datetime.now())) + str(
        datetime.time(datetime.now())) + "/"
    directory = os.path.dirname(output_directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # setup the logging to a file
    logging.basicConfig(filename=output_directory + 'log.log', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # Remove all the plt and PIL  debug messages
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    # Create the output directory where the 2-opt animation temporary files are saved
    output_directory_2_opt_animation = output_directory + "2-opt-animation/"
    directory = os.path.dirname(output_directory_2_opt_animation)
    if not os.path.exists(directory):
        os.makedirs(directory)

    output_directory_aco_animation = output_directory + "aco-animation/"
    directory = os.path.dirname(output_directory_aco_animation)
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

    clustered_data = None

    if program_options.CLUSTER_TYPE is ClusterAlgorithmType.K_MEANS:
        clustered_data = perform_k_means_clustering(problem_data_array, program_options)

    if program_options.CLUSTER_TYPE is ClusterAlgorithmType.AFFINITY_PROPAGATION:
        clustered_data = perform_affinity_propagation(problem_data_array, program_options)

    if program_options.CLUSTER_TYPE is ClusterAlgorithmType.BIRCH:
        clustered_data = perform_birch_clustering(problem_data_array, program_options)

    if program_options.CLUSTER_TYPE is ClusterAlgorithmType.DBSCAN:
        clustered_data = perform_dbscan_clustering(problem_data_array, program_options)

    if program_options.CLUSTER_TYPE is ClusterAlgorithmType.OPTICS:
        clustered_data = perform_optics_clustering(problem_data_array, program_options)

    # Set the overall node dicts onto the clustering object
    clustered_data.node_location_to_id_dict = node_location_to_id_dict
    clustered_data.node_id_to_location_dict = node_id_to_location_dict

    cluster_nodes = clustered_data.get_dict_node_id_location_mapping_aco()
    cluster_nodes_dict = clustered_data.get_dict_node_id_location_mapping_aco()
    logging.debug("%s nodes after clustering", len(cluster_nodes_dict))

    # Raise an error if only 1 cluster has come out of this because ACO needs more than 1 cluster to run over
    if len(cluster_nodes_dict) <= 1:
        raise ValueError("Need more than one cluster from the clustering algorithm")

    aco_tour_improvement_plotter: TourImprovementAnimator = TourImprovementAnimator(cluster_nodes_dict,
                                                                                    problem_type="aco")

    before = datetime.now()
    if program_options.ACO_TYPE is ACOType.ACO_MULTITHREADED:
        colony = AntColony(nodes=cluster_nodes_dict, distance_callback=aco_distance_callback,
                           alpha=program_options.ACO_ALPHA_VALUE,
                           beta=program_options.ACO_BETA_VALUE,
                           pheromone_evaporation_coefficient=program_options.ACO_RHO_VALUE,
                           pheromone_constant=program_options.ACO_Q_VALUE, ant_count=program_options.ACO_ANT_COUNT,
                           tour_improvement_animator=aco_tour_improvement_plotter,
                           iterations=program_options.ACO_ITERATIONS)
        answer = colony.mainloop()

    elif program_options.ACO_TYPE is ACOType.ACO_PY:
        solver = acopy.Solver(rho=program_options.ACO_RHO_VALUE, q=program_options.ACO_Q_VALUE)
        colony = acopy.Colony(alpha=program_options.ACO_ALPHA_VALUE, beta=program_options.ACO_BETA_VALUE)

        logger_plugin = LoggerPlugin()
        iteration_plotter_plugin = IterationPlotterPlugin(tour_improvement_animator=aco_tour_improvement_plotter)

        solver.add_plugin(logger_plugin)
        solver.add_plugin(iteration_plotter_plugin)

        graph = clustered_data.turn_clusters_into_nx_graph(problem)
        solution = solver.solve(graph, colony, limit=program_options.ACO_ITERATIONS,
                                gen_size=program_options.ACO_ANT_COUNT)
        answer = solution.nodes

    else:
        raise NotImplementedError()
    after = datetime.now()

    dif = after - before
    logging.debug("Time taken for %s aco %s", program_options.ACO_TYPE, dif)

    clustered_data.aco_cluster_tour = answer

    clustered_data.find_nodes_to_move_between_clusters()
    if program_options.ACO_TYPE is ACOType.ACO_MULTITHREADED:
        clustered_data.find_tours_within_clusters_using_multithreaded_aco()

    elif program_options.ACO_TYPE is ACOType.ACO_PY:
        clustered_data.find_tours_within_clusters_using_acopy()

    else:
        raise NotImplementedError()

    tour_node_coordinates = clustered_data.get_ordered_nodes_for_all_clusters()

    counter = 0
    tour_node_id = []

    for node in tour_node_coordinates:
        tour_node_id.append(node_location_to_id_dict[repr(node)])

    clustered_data.node_level_tour = tour_node_id

    tour_node_id_set = set(tour_node_id)
    valid = len(tour_node_id) == len(tour_node_id_set)
    logging.debug("Tour is valid %s", valid)

    length_before = calculate_distance_for_tour(tour_node_id, node_id_to_location_dict)
    logging.debug("Length before 2-opt is %s", length_before)

    # If the option to run 2opt is set then process 2-opt
    if program_options.SHOULD_RUN_2_OPT:
        tsp_2_opt_graph_animator = TourImprovementAnimator(node_id_to_location_dict, problem_type="2-opt")

        before = datetime.now()
        final_route = run_2_opt(existing_route=tour_node_id, node_id_to_location_dict=node_id_to_location_dict,
                                distance_calculator_callback=calculate_distance_for_tour,
                                tsp_2_opt_animator=tsp_2_opt_graph_animator)
        after = datetime.now()

        dif = after - before
        logging.debug("Time taken for 2-opt %s", dif)

        length_after = calculate_distance_for_tour(final_route, node_id_to_location_dict)
        logging.debug("Length after 2-opt is %s", length_after)

        logging.debug("Final route after 2-opt is %s", final_route)

    program_end_time = datetime.now()
    dif = program_end_time - program_start_time
    logging.debug("Time taken for entire program %s", dif)

    # These are the tour plotters so should be ignored for time calculations
    logging.debug("Starting tour plotters")

    # This is the graph that shows all the clusters
    plot_clustered_graph(tsp_problem_name, output_directory, colors, cluster_data=clustered_data,
                         cluster_type=str(program_options.CLUSTER_TYPE), display_plot=program_options.DISPLAY_PLOTS)

    # Plot all the nodes in the problem, no tour
    plot_nodes(problem_data_array, tsp_problem_name, output_directory, display_plot=program_options.DISPLAY_PLOTS)

    plot_aco_clustered_tour(answer, clustered_data, display_plot=program_options.DISPLAY_PLOTS,
                            tsp_problem_name=tsp_problem_name, output_directory=output_directory)

    # Plot the tour pre 2-opt
    plot_complete_tsp_tour(tour_node_id, node_id_to_location_dict,
                           title="TSP Tour Before 2-opt. Length: " + str(length_before),
                           tsp_problem_name=tsp_problem_name, output_directory=output_directory,
                           display_plot=program_options.DISPLAY_PLOTS)

    # If 2opt was ran then you can safely print out all the 2-opt related graphs
    if program_options.SHOULD_RUN_2_OPT:
        # Plot the tour post 2-opt
        plot_complete_tsp_tour(final_route, node_id_to_location_dict,
                               title="TSP Tour After 2-opt. Length: " + str(length_after),
                               tsp_problem_name=tsp_problem_name, output_directory=output_directory,
                               display_plot=program_options.DISPLAY_PLOTS)

        # Plot the tour post 2-opt with node ids printed
        plot_complete_tsp_tour(final_route, node_id_to_location_dict, title="Final TSP Tour With Node ID",
                               node_id_shown=True, tsp_problem_name=tsp_problem_name, output_directory=output_directory,
                               display_plot=program_options.DISPLAY_PLOTS)

        # Create an animation of the 2-opt incremental improvement
        tsp_2_opt_graph_animator.animate(tsp_problem_name, output_directory, output_directory_2_opt_animation)

    # Create an animation of the aco incremental improvement
    aco_tour_improvement_plotter.animate(tsp_problem_name, output_directory, output_directory_aco_animation)
    logging.debug("Finished tour plotters")
