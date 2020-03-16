from unittest import TestCase

from Loading import load_problem_into_np_array


class TestClusteredData(TestCase):
    def test_cluster_neighbour_movement(self):
        problem, problem_data_array = load_problem_into_np_array("testdata/world/dj38.tsp")

        self.assertCountEqual(problem_data_array.shape, (38, 2), "The shape of the numpy array is incorrect")
