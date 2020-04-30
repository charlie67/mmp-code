import logging
import os
import sys
from datetime import datetime
from itertools import cycle

import acopy
import tsplib95

from aco.aco_type_enum import ACOType
from aco.acopy.iteration_plotter_plugin import IterationPlotterPlugin
from aco.acopy.logger_plugin import LoggerPlugin
from aco.multithreaded.multi_threaded_ant_colony import AntColony
from clustering.cluster_type_enum import InternalClusterPathFinderType, ClusterType
from clustering.clustered_data import ClusteredData, Cluster
from clustering.clustering import plot_clustered_graph, perform_affinity_propagation, perform_k_means_clustering, \
    perform_birch_clustering, perform_dbscan_clustering, perform_optics_clustering
from clustering.clustering_algorithm_type_enum import ClusterAlgorithmType
from distance_calculation import calculate_distance_for_tour, aco_distance_callback
from options.options_holder import Options
from plotting.graph_plotting import plot_nodes, plot_aco_clustered_tour, plot_complete_tsp_tour
from plotting.tour_improvement_plotter import TourImprovementAnimator
from tsp_2_opt.tsp_2_opt_improver import run_2_opt


def setup_program_options_from_args(args) -> Options:
    # These are all required
    file_name = args.input
    output_directory = args.output
    cluster_type_name = args.cluster_type.upper()
    aco_type_name = args.aco_type.upper()
    cluster_tour_type_name = args.cluster_tour_type.upper()

    # The output directory needs to end with a '/'
    if not output_directory.endswith('/'):
        output_directory += "/"

    output_directory_2_opt_animation = output_directory + "2-opt-animation/"
    output_directory_aco_animation = output_directory + "aco-animation/"

    # Split the filename on the slashes to split in into file/folder parts and then get the last bit because that
    # will be the filename of the TSP problem
    tsp_problem_name = file_name.split('/')[-1]

    try:
        cluster_type = ClusterAlgorithmType[cluster_type_name]
    except KeyError:
        raise NotImplementedError("The cluster type passed in was not found")

    try:
        aco_type = ACOType[aco_type_name]
    except KeyError:
        raise NotImplementedError("The ACO type passed in was not found")

    try:
        cluster_tour_type = InternalClusterPathFinderType[cluster_tour_type_name]
    except KeyError:
        raise NotImplementedError("The internal cluster tour finder type passed in was not found")

    # The optional arguments
    number_clusters = args.numberclusters if args.numberclusters else None

    # Affinity Propagation
    ap_convergence_iter = args.ap_convergence_iter if args.ap_convergence_iter else None
    ap_max_iter = args.ap_max_iter if args.ap_max_iter else None

    # Optics
    optics_min_samples = args.optics_min_samples if args.optics_min_samples else None

    # K_Means
    k_means_n_init = args.k_means_n_init if args.k_means_n_init else None

    # Birch
    birch_branching_factor = args.birch_branching_factor if args.birch_branching_factor else None
    birch_threshold = args.birch_threshold if args.birch_threshold else None

    # DBSCAN
    dbscan_eps = args.dbscan_eps if args.dbscan_eps else None
    dbscan_min_samples = args.dbscan_min_samples if args.dbscan_min_samples else None
    automate_dbscan_eps = args.automate_dbscan_eps if args.automate_dbscan_eps is not None else None

    # ACO parameters
    aco_alpha_value = args.aco_alpha_value if args.aco_alpha_value else None
    aco_beta_value = args.aco_beta_value if args.aco_beta_value else None
    aco_rho_value = args.aco_rho_value if args.aco_rho_value else None
    aco_q_value = args.aco_q_value if args.aco_q_value else None
    aco_ant_count = args.aco_ant_count if args.aco_ant_count else None
    aco_iterations = args.aco_iterations if args.aco_iterations else None

    # should 2-opt be ran
    should_run_2_opt = args.run2opt if args.run2opt is not None else None

    # Should the data be clustered
    should_cluster = args.should_cluster if args.should_cluster is not None else None

    # Should the plots be opened in a window
    display_plots = args.displayplots if args.displayplots is not None else None

    # mat plot lib dpi value
    plt_dpi_value = args.dpi if args.dpi else None

    # Create the ouput directory where all the graphs are saved
    directory = os.path.dirname(output_directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Create the output directory where the 2-opt animation temporary files are saved
    directory = os.path.dirname(output_directory_2_opt_animation)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Create the output directory where the aco animation temporary files are saved
    directory = os.path.dirname(output_directory_aco_animation)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # setup the logging to a file
    logging.basicConfig(filename=output_directory + 'log.log', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # Remove all the plt and PIL  debug messages
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    program_options = Options(output_directory=output_directory, tsp_problem_name=tsp_problem_name, file_name=file_name,
                              output_directory_aco_animation=output_directory_aco_animation,
                              output_directory_2_opt_animation=output_directory_2_opt_animation,
                              number_clusters=number_clusters, display_plots=display_plots, aco_type=aco_type,
                              cluster_type=cluster_type, cluster_tour_type=cluster_tour_type,
                              affinity_propagation_convergence_iterations=ap_convergence_iter,
                              affinity_propagation_max_iterations=ap_max_iter,
                              optics_min_samples=optics_min_samples, k_means_n_init=k_means_n_init,
                              birch_branching_factor=birch_branching_factor, birch_threshold=birch_threshold,
                              dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples,
                              automate_dbscan_eps=automate_dbscan_eps, aco_alpha_value=aco_alpha_value,
                              aco_beta_value=aco_beta_value, aco_rho_value=aco_rho_value, aco_q_value=aco_q_value,
                              aco_ant_count=aco_ant_count, aco_iterations=aco_iterations,
                              should_run_2_opt=should_run_2_opt, should_cluster=should_cluster,
                              plt_dpi_value=plt_dpi_value)
    return program_options


def run_algorithm_with_options(program_options: Options, problem_data_array, problem: tsplib95.Problem):
    program_start_time = datetime.now()

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
    if program_options.SHOULD_CLUSTER:
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
    else:
        clustered_data = ClusteredData(nodes=problem_data_array, clusters=list, program_options=program_options)

        for node in problem_data_array:
            cluster = Cluster(cluster_centre=node, nodes=[node], cluster_type=ClusterType.UNCLASSIFIED_NODE_CLUSTER,
                              program_options=program_options)
            clustered_data.add_unclassified_node(cluster)

    # Set the overall node dicts onto the clustering object
    clustered_data.node_location_to_id_dict = node_location_to_id_dict
    clustered_data.node_id_to_location_dict = node_id_to_location_dict
    cluster_nodes_dict = clustered_data.get_dict_node_id_location_mapping_aco()

    logging.debug("%s nodes after clustering", len(cluster_nodes_dict))

    # Raise an error if only 1 cluster has come out of this because ACO needs more than 1 cluster to run over
    if len(cluster_nodes_dict) <= 1:
        raise ValueError("Need more than one cluster from the clustering algorithm")

    aco_tour_improvement_plotter: TourImprovementAnimator = TourImprovementAnimator(cluster_nodes_dict,
                                                                                    problem_type="aco",
                                                                                    program_options=program_options)
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

    logging.debug("Time taken for initial global %s aco %s", program_options.ACO_TYPE, dif)

    clustered_data.aco_cluster_tour = answer
    clustered_data.find_nodes_to_move_between_clusters()

    if program_options.CLUSTER_TOUR_TYPE is InternalClusterPathFinderType.ACO:
        if program_options.ACO_TYPE is ACOType.ACO_MULTITHREADED:
            clustered_data.find_tours_within_clusters_using_multithreaded_aco()
        elif program_options.ACO_TYPE is ACOType.ACO_PY:
            clustered_data.find_tours_within_clusters_using_acopy()
        else:
            raise NotImplementedError()

    elif program_options.CLUSTER_TOUR_TYPE is InternalClusterPathFinderType.GREEDY_NEAREST_NODE:
        clustered_data.find_tours_within_clusters_using_greedy_closest_nodes()
    else:
        raise NotImplementedError()
    tour_node_coordinates = clustered_data.get_ordered_nodes_for_all_clusters()

    # Tour as node ids instead of node locations
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
        tsp_2_opt_graph_animator = TourImprovementAnimator(node_id_to_location_dict, problem_type="2-opt",
                                                           program_options=program_options)

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

    # plot the tours for each cluster
    clustered_data.plot_all_cluster_tours()

    # This is the graph that shows all the clusters
    plot_clustered_graph(colors, cluster_data=clustered_data, program_options=program_options)

    # Plot all the nodes in the problem, no tour
    plot_nodes(problem_data_array, program_options=program_options)

    # Plot the ACO tour of the clusters
    plot_aco_clustered_tour(answer, clustered_data, program_options=program_options)

    # Plot the tour pre 2-opt
    plot_complete_tsp_tour(tour_node_id, node_id_to_location_dict,
                           title="TSP Tour Before 2-opt. Length: " + str(length_before),
                           program_options=program_options)

    # If 2opt was ran then you can safely print out all the 2-opt related graphs
    if program_options.SHOULD_RUN_2_OPT:
        # Plot the tour post 2-opt
        plot_complete_tsp_tour(final_route, node_id_to_location_dict,
                               title="TSP Tour After 2-opt. Length: " + str(length_after),
                               program_options=program_options)

        # Plot the tour post 2-opt with node ids printed
        plot_complete_tsp_tour(final_route, node_id_to_location_dict, title="Final TSP Tour With Node ID",
                               node_id_shown=True, program_options=program_options)

        # Create an animation of the 2-opt incremental improvement
        tsp_2_opt_graph_animator.animate(
            output_directory_animation_graphs=program_options.OUTPUT_DIRECTORY_2_OPT_ANIMATION)

    # Create an animation of the aco incremental improvement
    aco_tour_improvement_plotter.animate(
        output_directory_animation_graphs=program_options.OUTPUT_DIRECTORY_ACO_ANIMATION)

    logging.debug("Finished tour plotters")
