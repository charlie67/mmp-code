# source: http://stackoverflow.com/a/11158224/5343977
import inspect
import os
import sys
import unittest

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from aco.multithreaded import multi_threaded_ant_colony as module


class TestAntColonyInitNodes(unittest.TestCase):
    def test_correct(self):
        module.debug = False

        # setup
        class test_empty_object(module.AntColony):
            def __init__(self): pass

            def _get_distance(self, start, end): pass

            def _init_matrix(self, size, value=None): pass

            def _init_ants(self, start=0): pass

            def _update_pheromone_map(self): pass

            def _populate_ant_updated_pheromone_map(self, ant): pass

            def mainloop(self): pass

        # def _init_nodes(self, nodes): pass

        test_object = test_empty_object()

        testing_nodes = {
            10: (1, 1),
            15: (0, 0),
            17: (2, 2),
            20: (3, 3),
        }

        # testing
        id_to_key, id_to_values = test_object._init_nodes(testing_nodes)

        # import debug
        # debug.DEBUG(id_to_key)
        # debug.DEBUG(id_to_values)

        self.assertEqual({0: 10, 1: 15, 2: 17, 3: 20}, id_to_key)
        self.assertEqual({0: (1, 1), 1: (0, 0), 2: (2, 2), 3: (3, 3)}, id_to_values)


if __name__ == '__main__':
    unittest.main()
