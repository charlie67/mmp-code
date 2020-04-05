from unittest import TestCase

from clustering.cluster_type_enum import ClusterType
from clustering.clustered_data import ClusteredData, Cluster
from loading import load_problem_into_np_array
from clustering.clustering import perform_affinity_propagation

import numpy as np


class TestClusteredData(TestCase):
    def test_cluster_neighbour_movement(self):
        # create a cluster data object and then run the method over it to test that it correctly calculates the movement
        problem, problem_data_array = load_problem_into_np_array("testdata/world/dj38.tsp")
        clustered_data = perform_affinity_propagation(problem_data_array)

        tour = []

        for i in range(len(clustered_data.clusters)):
            tour.append(i)

        clustered_data.aco_cluster_tour = tour

        clustered_data.find_nodes_to_move_between_clusters()
        all_clusters_to_check = clustered_data.get_all_clusters()

        print("all clusters to check", all_clusters_to_check)

        for cluster in all_clusters_to_check:
            # Check that the entry/exit nodes are not the same
            print("Length is", len(cluster.entry_exit_nodes), cluster.nodes)
            self.assertTrue(len(cluster.entry_exit_nodes) == 2)
            self.assertFalse(cluster.entry_exit_nodes[0] == cluster.entry_exit_nodes[1])

        clustered_data.find_tours_within_clusters_using_closest_nodes()
        all_clusters_to_check = clustered_data.get_all_clusters()
        for cluster in all_clusters_to_check:
            # Check that the entry/exit nodes are not the same
            self.assertTrue(len(cluster.tour) == len(cluster.nodes))
            self.assertFalse(cluster.entry_exit_nodes[0] == cluster.entry_exit_nodes[1])

    def test_cluster_creation_and_retrieval(self):
        # Create a clustered data object with no nodes or clusters just empty lists
        clustered_data = ClusteredData(list(), list())

        for i in range(5):
            centre = np.zeros(2)
            centre[0] = i
            centre[1] = i
            cluster = Cluster(cluster_centre=centre, nodes=centre, cluster_type=ClusterType.UNCLASSIFIED_NODE_CLUSTER)

            clustered_data.add_unclassified_node(cluster)

        self.assertEqual(len(clustered_data.get_all_clusters()), 5)
        self.assertEqual(len(clustered_data.get_all_cluster_centres_and_unclassified_node_locations()), 5)

        for i in range(3):
            nodes = []
            for j in range(3):
                node = np.zeros(2)
                node[0] = i * 2
                node[1] = i * 2
                nodes.append(node)

            cluster_centre = np.zeros(2)
            cluster_centre[0] = i * 2
            cluster_centre[1] = i * 2

            cluster = Cluster(cluster_centre=cluster_centre, nodes=nodes, cluster_type=ClusterType.FULL_CLUSTER)
            clustered_data.add_cluster(cluster)

        self.assertEqual(len(clustered_data.get_all_clusters()), 8)
        self.assertEqual(len(clustered_data.get_all_cluster_centres_and_unclassified_node_locations()), 8)
