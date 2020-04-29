from unittest import TestCase

from loading import load_problem_into_np_array


class TestClusteredData(TestCase):
    def test_cluster_neighbour_movement(self):
        # Load the DJ38 file
        problem, problem_data_array = load_problem_into_np_array("testfile.tsp")

        self.assertCountEqual(problem_data_array.shape, (38, 2), "The shape of the numpy array is incorrect")
