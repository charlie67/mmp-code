import tsplib95
import numpy as np


def load_problem_into_np_array(file_name):
    problem: tsplib95.models.Problem = tsplib95.utils.load_problem(file_name)
    problem_data_array = np.zeros(shape=(problem.dimension, 2))
    # transform the problem data into a numpy array of node coordinates for scikit learn to use
    for node in problem.get_nodes():
        problem_data_array[node - 1, 0] = problem.get_display(node)[1]
        problem_data_array[node - 1, 1] = problem.get_display(node)[0]

    return problem, problem_data_array
