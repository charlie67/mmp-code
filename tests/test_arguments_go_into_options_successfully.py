from argparse import Namespace
from unittest import TestCase

from aco.aco_type_enum import ACOType
from clustering.cluster_type_enum import InternalClusterPathFinderType
from clustering.clustering_algorithm_type_enum import ClusterAlgorithmType
from options import default_options
from runner import setup_program_options_from_args


class TestOptionsArgumentRetrieval(TestCase):
    def test_default_options_only_work(self):
        fake_args = Namespace(input="test/testinputfile", output="test", cluster_type="k_means", aco_type="aco_py",
                              numberclusters=None, cluster_tour_type="aco",
                              affinity_propagation_convergence_iterations=None,
                              affinity_propagation_max_iterations=None, optics_min_samples=None,
                              k_means_n_init=None, birch_branching_factor=None, birch_threshold=None, dbscan_eps=None,
                              dbscan_min_samples=None, automate_dbscan_eps=None, aco_alpha_value=None,
                              aco_beta_value=None, aco_rho_value=None, aco_q_value=None, aco_ant_count=None,
                              aco_iterations=None, run2opt=None, should_cluster=None, displayplots=None, dpi=None)

        program_options = setup_program_options_from_args(fake_args)

        self.assertEqual("test/testinputfile", program_options.FILE_NAME)
        self.assertEqual("testinputfile", program_options.TSP_PROBLEM_NAME)

        self.assertEqual(ClusterAlgorithmType.K_MEANS, program_options.CLUSTER_TYPE)

        self.assertEqual(ACOType.ACO_PY, program_options.ACO_TYPE)
        self.assertEqual(InternalClusterPathFinderType.ACO, program_options.CLUSTER_TOUR_TYPE)

        self.assertEqual(default_options.NUMBER_CLUSTERS, program_options.NUMBER_CLUSTERS)

        self.assertEqual(default_options.AFFINITY_PROPAGATION_CONVERGENCE_ITERATIONS,
                         program_options.AFFINITY_PROPAGATION_CONVERGENCE_ITERATIONS)
        self.assertEqual(default_options.AFFINITY_PROPAGATION_MAX_ITERATIONS,
                         program_options.AFFINITY_PROPAGATION_MAX_ITERATIONS)

        self.assertEqual(default_options.OPTICS_MIN_SAMPLES, program_options.OPTICS_MIN_SAMPLES)

        self.assertEqual(default_options.K_MEANS_N_INIT, program_options.K_MEANS_N_INIT)

        self.assertEqual(default_options.BIRCH_BRANCHING_FACTOR, program_options.BIRCH_BRANCHING_FACTOR)
        self.assertEqual(default_options.BIRCH_THRESHOLD, program_options.BIRCH_THRESHOLD)

        self.assertEqual(default_options.DBSCAN_EPS, program_options.DBSCAN_EPS)
        self.assertEqual(default_options.DBSCAN_MIN_SAMPLES, program_options.DBSCAN_MIN_SAMPLES)
        self.assertEqual(default_options.AUTOMATE_DBSCAN_EPS, program_options.AUTOMATE_DBSCAN_EPS)

        self.assertEqual(default_options.ACO_ALPHA_VALUE, program_options.ACO_ALPHA_VALUE)
        self.assertEqual(default_options.ACO_BETA_VALUE, program_options.ACO_BETA_VALUE)
        self.assertEqual(default_options.ACO_RHO_VALUE, program_options.ACO_RHO_VALUE)
        self.assertEqual(default_options.ACO_Q_VALUE, program_options.ACO_Q_VALUE)
        self.assertEqual(default_options.ACO_ANT_COUNT, program_options.ACO_ANT_COUNT)
        self.assertEqual(default_options.ACO_ITERATIONS, program_options.ACO_ITERATIONS)

        self.assertEqual(default_options.SHOULD_RUN_2_OPT, program_options.SHOULD_RUN_2_OPT)
        self.assertEqual(default_options.SHOULD_CLUSTER, program_options.SHOULD_CLUSTER)
        self.assertEqual(default_options.DISPLAY_PLOTS, program_options.DISPLAY_PLOTS)
        self.assertEqual(default_options.PLT_DPI_VALUE, program_options.PLT_DPI_VALUE)

    def test_all_options_work(self):
        fake_args = Namespace(input="test/testinputfile", output="test", cluster_type="k_means", aco_type="aco_py",
                              numberclusters=1000, cluster_tour_type="aco",
                              ap_convergence_iter=999, ap_max_iter=765, optics_min_samples=5,
                              k_means_n_init=3, birch_branching_factor=45, birch_threshold=67, dbscan_eps=43.23,
                              dbscan_min_samples=3, automate_dbscan_eps=True, aco_alpha_value=12,
                              aco_beta_value=4, aco_rho_value=13, aco_q_value=14, aco_ant_count=15,
                              aco_iterations=16, run2opt=False, should_cluster=False, displayplots=True, dpi=9000)

        program_options = setup_program_options_from_args(fake_args)

        self.assertEqual("test/testinputfile", program_options.FILE_NAME)
        self.assertEqual("testinputfile", program_options.TSP_PROBLEM_NAME)

        self.assertEqual(ClusterAlgorithmType.K_MEANS, program_options.CLUSTER_TYPE)

        self.assertEqual(ACOType.ACO_PY, program_options.ACO_TYPE)
        self.assertEqual(InternalClusterPathFinderType.ACO, program_options.CLUSTER_TOUR_TYPE)

        self.assertEqual(1000, program_options.NUMBER_CLUSTERS)

        self.assertEqual(999, program_options.AFFINITY_PROPAGATION_CONVERGENCE_ITERATIONS)
        self.assertEqual(765, program_options.AFFINITY_PROPAGATION_MAX_ITERATIONS)

        self.assertEqual(5, program_options.OPTICS_MIN_SAMPLES)

        self.assertEqual(3, program_options.K_MEANS_N_INIT)

        self.assertEqual(45, program_options.BIRCH_BRANCHING_FACTOR)
        self.assertEqual(67, program_options.BIRCH_THRESHOLD)

        self.assertEqual(43.23, program_options.DBSCAN_EPS)
        self.assertEqual(3, program_options.DBSCAN_MIN_SAMPLES)
        self.assertEqual(True, program_options.AUTOMATE_DBSCAN_EPS)

        self.assertEqual(12, program_options.ACO_ALPHA_VALUE)
        self.assertEqual(4, program_options.ACO_BETA_VALUE)
        self.assertEqual(13, program_options.ACO_RHO_VALUE)
        self.assertEqual(14, program_options.ACO_Q_VALUE)
        self.assertEqual(15, program_options.ACO_ANT_COUNT)
        self.assertEqual(16, program_options.ACO_ITERATIONS)

        self.assertEqual(False, program_options.SHOULD_RUN_2_OPT)
        self.assertEqual(False, program_options.SHOULD_CLUSTER)
        self.assertEqual(True, program_options.DISPLAY_PLOTS)
        self.assertEqual(9000, program_options.PLT_DPI_VALUE)