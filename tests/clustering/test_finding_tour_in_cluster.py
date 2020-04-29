import unittest
from unittest import TestCase

import numpy as np

from clustering.cluster_type_enum import ClusterType
from clustering.clustered_data import ClusteredData, Cluster
from options.options_holder import Options


class TestClusteredData(TestCase):
    def test_finding_cluster_tour_with_nearest_neighbours(self):
        clustered_data, test_nodes = self.setup_cluster()

        clustered_data.find_nodes_to_move_between_clusters()

        clustered_data.find_tours_within_clusters_using_greedy_closest_nodes()

        self.tour_is_valid_checker(clustered_data, test_nodes)

    def test_finding_cluster_tour_with_multithreaded_aco(self):
        clustered_data, test_nodes = self.setup_cluster()

        clustered_data.find_nodes_to_move_between_clusters()

        clustered_data.find_tours_within_clusters_using_multithreaded_aco()

        self.tour_is_valid_checker(clustered_data, test_nodes)

    def test_finding_cluster_tour_with_acopy(self):
        clustered_data, test_nodes = self.setup_cluster()

        clustered_data.find_nodes_to_move_between_clusters()

        clustered_data.find_tours_within_clusters_using_acopy()

        self.tour_is_valid_checker(clustered_data, test_nodes)

    def tour_is_valid_checker(self, clustered_data, test_nodes):
        # The tour in coordinate form
        tour_node_coordinates = clustered_data.get_ordered_nodes_for_all_clusters()
        valid = len(tour_node_coordinates) == len(test_nodes)
        self.assertTrue(valid)
        # For each cluster need to ensure that it has 2 entry exit nodes which are not the same and that it has a tour
        # going from the start to the end
        for cluster in clustered_data.get_all_clusters():
            if cluster.cluster_type is ClusterType.FULL_CLUSTER:
                entry_exit_nodes = cluster.entry_exit_nodes
                tour = cluster.tour

                self.assertNotEqual(entry_exit_nodes[0], entry_exit_nodes[1])
                self.assertEqual(entry_exit_nodes[0], tour[0])
                self.assertEqual(entry_exit_nodes[1], tour[-1])
            else:
                # This is an UNCLASSIFIED_NODE_CLUSTER and therefore only has one node
                entry_exit_nodes = cluster.entry_exit_nodes

                self.assertEqual(1, len(cluster.nodes))
                self.assertEqual(entry_exit_nodes[0], entry_exit_nodes[1])

    @staticmethod
    def setup_cluster():
        test_program_options = Options(output_directory="test", tsp_problem_name="test", aco_ant_count=10,
                                       aco_iterations=5)
        # This data has three clusters one big one that has the first four nodes and two other clusters of size one
        # that have only 1 node in
        test_nodes = [[1, 1], [1.5, 1], [1, 1.5], [1, 2], [2, 1], [2, 2]]
        # need to turn this list into an np array
        nodes = np.asarray(test_nodes)
        clustered_data: ClusteredData = ClusteredData(nodes=nodes, clusters=list(),
                                                      program_options=test_program_options)
        main_cluster = Cluster(cluster_centre=[1.25, 1.5], nodes=np.asarray([[1, 1], [1.5, 1], [1, 1.5], [1, 2]]),
                               cluster_type=ClusterType.FULL_CLUSTER, program_options=test_program_options)
        clustered_data.add_cluster(main_cluster)
        unclassified_node = Cluster(cluster_centre=[2, 1], nodes=np.asarray([[2, 1]]),
                                    cluster_type=ClusterType.UNCLASSIFIED_NODE_CLUSTER,
                                    program_options=test_program_options)
        clustered_data.add_unclassified_node(unclassified_node)
        unclassified_node = Cluster(cluster_centre=[2, 2], nodes=np.asarray([[2, 2]]),
                                    cluster_type=ClusterType.UNCLASSIFIED_NODE_CLUSTER,
                                    program_options=test_program_options)
        clustered_data.add_unclassified_node(unclassified_node)
        clustered_data.aco_cluster_tour = (0, 1, 2)
        return clustered_data, test_nodes
