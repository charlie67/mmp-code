from unittest import TestCase

from Loading import load_problem_into_np_array
from Clustering import perform_affinity_propagation


class TestClusteredData(TestCase):
    def test_cluster_neighbour_movement(self):
        # create a cluster data object and then run the method over it to test that it correctly calculates the movement
        problem, problem_data_array = load_problem_into_np_array("testdata/world/dj38.tsp")
        clustered_data = perform_affinity_propagation(problem_data_array)

        clustered_data.aco_cluster_tour = range(len(clustered_data.clusters))

        clustered_data.find_nodes_to_move_between_clusters()
        all_clusters_to_check = clustered_data.get_all_clusters()

        for cluster in all_clusters_to_check:
            # Check that the entry/exit nodes are not the same
            self.assertEqual(len(cluster.entry_exit_nodes), 2)
            self.assertFalse(cluster.entry_exit_nodes[0] == cluster.entry_exit_nodes[1])


