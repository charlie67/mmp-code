import logging
import multiprocessing

import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors

from options.options_holder import Options


def plot(data, title, file_name, display_plots):
    plt.plot(data)
    plt.title(title)
    plt.savefig(file_name)
    if display_plots:
        plt.show()
    plt.close()


def find_using_nearest_neighbours(problem_data_array, program_options: Options):
    processes = []

    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(problem_data_array)
    distances, indices = nbrs.kneighbors(problem_data_array)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]

    process = multiprocessing.Process(target=plot, args=(distances, "The distance density distribution",
                                                         program_options.OUTPUT_DIRECTORY + "auto_eps_value_distance_density.png",
                                                         program_options.DISPLAY_PLOTS))
    processes.append(process)

    gradient = np.gradient(distances)
    process = multiprocessing.Process(target=plot, args=(
        gradient, "Gradient of the distance graph", program_options.OUTPUT_DIRECTORY + "auto_eps_value_gradient.png",
        program_options.DISPLAY_PLOTS))
    processes.append(process)

    end_num = int(len(distances) * 6 / 7)
    distances = distances[:end_num]
    process = multiprocessing.Process(target=plot, args=(
        distances, "6/7 of the distance matrix", program_options.OUTPUT_DIRECTORY + "auto_eps_value_cut_graph.png",
        program_options.DISPLAY_PLOTS))
    processes.append(process)

    for process in processes:
        process.start()

    new_eps_value = distances[-1]
    logging.debug("new eps value is " + str(new_eps_value))
    return new_eps_value
