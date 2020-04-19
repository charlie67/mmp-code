from aco.aco_type_enum import ACOType
from clustering.clustering_algorithm_type_enum import ClusterAlgorithmType
from options import default_options


class Options:

    def __init__(self, number_clusters=None, display_plots=None, aco_type=None, cluster_type=None,
                 affinity_propagation_convergence_iterations=None, affinity_propagation_max_iterations=None,
                 optics_min_samples=None, k_means_n_init=None, birch_branching_factor=None, birch_threshold=None,
                 dbscan_eps=None, dbscan_min_samples=None, aco_alpha_value=None, aco_beta_value=None,
                 aco_rho_value=None, aco_q_value=None, aco_ant_count=None, aco_iterations=None, should_run_2_opt=None,
                 should_cluster=None) -> None:
        super().__init__()
        self.SHOULD_CLUSTER = should_cluster if should_cluster is not None else default_options.SHOULD_CLUSTER
        self.SHOULD_RUN_2_OPT = should_run_2_opt if should_run_2_opt is not None else default_options.SHOULD_RUN_2_OPT

        self.ACO_ITERATIONS = aco_iterations if aco_iterations is not None else default_options.ACO_ITERATIONS
        self.ACO_ANT_COUNT = aco_ant_count if aco_ant_count is not None else default_options.ACO_ANT_COUNT
        self.ACO_Q_VALUE = aco_q_value if aco_q_value is not None else default_options.ACO_Q_VALUE
        self.ACO_RHO_VALUE = aco_rho_value if aco_rho_value is not None else default_options.ACO_RHO_VALUE
        self.ACO_BETA_VALUE = aco_beta_value if aco_beta_value is not None else default_options.ACO_BETA_VALUE
        self.ACO_ALPHA_VALUE = aco_alpha_value if aco_alpha_value is not None else default_options.ACO_ALPHA_VALUE
        self.ACO_TYPE = aco_type if aco_type is not None else default_options.ACO_TYPE

        self.DBSCAN_MIN_SAMPLES = dbscan_min_samples if dbscan_min_samples is not None else default_options.DBSCAN_MIN_SAMPLES
        self.DBSCAN_EPS = dbscan_eps if dbscan_eps is not None else default_options.DBSCAN_EPS

        self.BIRCH_THRESHOLD = birch_threshold if birch_threshold is not None else default_options.BIRCH_THRESHOLD
        self.BIRCH_BRANCHING_FACTOR = birch_branching_factor if birch_branching_factor is not None else default_options.BIRCH_BRANCHING_FACTOR

        self.NUMBER_CLUSTERS = number_clusters if number_clusters is not None else default_options.NUMBER_CLUSTERS
        self.K_MEANS_N_INIT = k_means_n_init if k_means_n_init is not None else default_options.K_MEANS_N_INIT

        self.OPTICS_MIN_SAMPLES = optics_min_samples if optics_min_samples is not None else default_options.OPTICS_MIN_SAMPLES

        self.AFFINITY_PROPAGATION_MAX_ITERATIONS = affinity_propagation_max_iterations if \
            affinity_propagation_max_iterations is not None else default_options.AFFINITY_PROPAGATION_MAX_ITERATIONS

        self.AFFINITY_PROPAGATION_CONVERGENCE_ITERATIONS = affinity_propagation_convergence_iterations if \
            affinity_propagation_convergence_iterations is not None else default_options.AFFINITY_PROPAGATION_CONVERGENCE_ITERATIONS

        self.CLUSTER_TYPE = cluster_type if cluster_type is not None else default_options.CLUSTER_TYPE

        self.DISPLAY_PLOTS = display_plots if display_plots is not None else default_options.DISPLAY_PLOTS
