import argparse

from loading import load_problem_into_np_array
from runner import run_algorithm_with_options, setup_program_options_from_args

if __name__ == '__main__':
    # From the arguments tsp_problem_name cluster_type and the output_directory need to be specified
    parser = argparse.ArgumentParser(description='Solve TSPs using ACO and clustering')

    parser.add_argument('-i', '--input', type=str, help='The file location for the inputfile')
    parser.add_argument('-o', '--output', type=str, help='The output folder that everything should be saved to')

    parser.add_argument('-c', '--cluster_type', type=str,
                        help='The clustering type to use, options are K_MEANS, AFFINITY_PROPAGATION, BIRCH, DBSCAN, OPTICS')
    parser.add_argument('-aco', '--aco_type', type=str,
                        help='The ACO algorithm that will be used, options are ACO_PY or ACO_MULTITHREADED')
    parser.add_argument('-ctt', '--cluster_tour_type', type=str,
                        help='The value to use for cluster_tour_type. The options are ACO or GREEDY_NEAREST_NODE. If ACO is selected then the ACO algorithm specified in -aco is used')

    # These are all optional arguments
    parser.add_argument('-n', '--numberclusters', type=int,
                        help='The number of clusters that should be created, this is only applied when the clustering '
                             'algorithm requires the number of clusters to be set')

    # Affinity Propagation clustering parameters
    parser.add_argument('-apci', '--ap_convergence_iter', type=int,
                        help='The value to use for affinity_propagation_convergence_iterations only used when Affinity Propagation clustering is selected')
    parser.add_argument('-apmi', '--ap_max_iter', type=int,
                        help='The value to use for affinity_propagation_max_iterations only used when Affinity Propagation clustering is selected')

    # Optics clustering parameters
    parser.add_argument('-oms', '--optics_min_samples', type=int,
                        help='The value to use for optics_min_samples only used when Optics clustering is selected')

    # K-Means clustering parameters
    parser.add_argument('-kmni', '--k_means_n_init', type=int,
                        help='The value to use for k_means_n_init only used when K_Means clustering is selected')

    # Birch clustering parameters
    parser.add_argument('-bbf', '--birch_branching_factor', type=int,
                        help='The value to use for birch_branching_factor only used when Birch clustering is selected')
    parser.add_argument('-bt', '--birch_threshold', type=float,
                        help='The value to use for birch_threshold only used when Birch clustering is selected')

    # DBSCAN
    parser.add_argument('-de', '--dbscan_eps', type=float,
                        help='The value to use for dbscan_eps only used when DBSCAN clustering is selected')
    parser.add_argument('-dms', '--dbscan_min_samples', type=int,
                        help='The value to use for dbscan_min_samples only used when DBSCAN clustering is selected')
    parser.add_argument('-ade', '--automate_dbscan_eps', type=str,
                        help='The value to use for automate_dbscan_eps only used when DBSCAN clustering is selected')

    # ACO parameter values
    parser.add_argument('-aav', '--aco_alpha_value', type=float, help='The value to use for aco_alpha_value')
    parser.add_argument('-abv', '--aco_beta_value', type=float, help='The value to use for aco_beta_value')
    parser.add_argument('-arv', '--aco_rho_value', type=float, help='The value to use for aco_rho_value')
    parser.add_argument('-aqv', '--aco_q_value', type=float, help='The value to use for aco_q_value')
    parser.add_argument('-aac', '--aco_ant_count', type=int, help='The value to use for aco_ant_count')
    parser.add_argument('-ai', '--aco_iterations', type=int, help='The value to use for aco_iterations')

    # Should 2-opt be ran
    parser.add_argument('-run2opt', type=str, help='Should 2-opt be ran')

    # Should the data be clustered
    parser.add_argument('-sc', '--should_cluster', type=str, help='Should the data be clustered')

    # The DPI value to use for matplotlib
    parser.add_argument('-dpi', type=int,
                        help='The value to use for plt_dpi_value. This effects the size of the output graphs')

    # Should the plots be displayed in a window
    parser.add_argument('-dp', '--displayplots', type=str,
                        help="Should the mat plot lib plots open up in a new window. They are always saved regardless of this choice")

    args = parser.parse_args()

    program_options = setup_program_options_from_args(args)

    problem, problem_data_array = load_problem_into_np_array(program_options.FILE_NAME)
    run_algorithm_with_options(program_options, problem_data_array, problem)











