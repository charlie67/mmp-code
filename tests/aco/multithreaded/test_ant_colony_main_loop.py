# source: http://stackoverflow.com/a/11158224/5343977
import inspect
import os
import sys
import unittest

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from aco.multithreaded import multi_threaded_ant_colony as module


class TestAntColonyMainLoop(unittest.TestCase):
    def test_correct(self):
        module.debug = False

        class test_empty_object(module.AntColony):
            def __init__(self): pass

            def _get_distance(self, start, end): return 1

            def _init_matrix(self, size, value=None): pass

            def _init_ants(self, start): pass

            def _update_pheromone_map(self): pass

            def _populate_ant_updated_pheromone_map(self, ant): pass

            def distance_callback(self): pass

        # def mainloop(self): pass
        test_object = test_empty_object()
        test_object.tour_improvement_animator = None

        # setup test environment
        def _init_matrix(size, value=None):
            """
            setup a matrix NxN (where n = size)
            used in both self.distance_matrix and self.pheromone_map
            as they require identical matrixes besides which value to initialize to
            """
            ret = []
            for row in range(size):
                ret.append([value for x in range(size)])
            return ret

        class mock_ant:
            def get_distance_traveled(self): return 1

            def get_route(self): return [0]

            def start(self): pass

            def join(self): pass

        test_object.ants = [mock_ant()]
        self.called_populate_ant_updated_pheromone_map = False

        def mock_populate_ant_updated_pheromone_map(ant):
            self.called_populate_ant_updated_pheromone_map = True

        test_object._populate_ant_updated_pheromone_map = mock_populate_ant_updated_pheromone_map

        self.called_update_pheromone_map = False

        def mock_update_pheromone_map():
            self.called_update_pheromone_map = True

        test_object._update_pheromone_map = mock_update_pheromone_map

        test_object.iterations = 1

        test_object.first_pass = 10
        test_object.nodes = dict()
        test_object.shortest_distance = 10
        test_object.shortest_path_seen = [0]

        test_object.start = 0

        test_object.id_to_key = {0: 0}

        # testing
        test_object.mainloop()
        self.assertTrue(self.called_update_pheromone_map)
        self.assertTrue(self.called_populate_ant_updated_pheromone_map)

        # cleanup
        del self.called_update_pheromone_map
        del self.called_populate_ant_updated_pheromone_map


if __name__ == '__main__':
    unittest.main()
