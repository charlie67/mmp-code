import multiprocessing
import os
import time

import imageio
import matplotlib.pyplot as plt

from options.options_holder import Options


def plot(tour, file_name, node_id_to_coordinate_dict):
    num = 0
    figure = plt.figure()
    for i in tour:
        j = tour[num - 1]

        node_i = node_id_to_coordinate_dict[i]
        node_j = node_id_to_coordinate_dict[j]

        plt.plot(node_i[0], node_i[1], 'b.', markersize=10, figure=figure)
        plt.plot([node_i[0], node_j[0]], [node_i[1], node_j[1]], 'k', linewidth=0.5, figure=figure)
        num += 1

    plt.savefig(file_name, dpi=200)
    plt.clf()
    plt.close(figure)


class TourImprovementAnimator:
    tour_history: list

    node_id_to_coordinate_dict: dict

    # The type of problem (aco, tsp) that this is animating, appended to the output file name
    problem_type: str

    def __init__(self, node_id_to_coordinate_dict, problem_type, program_options: Options) -> None:
        self.tour_history = list()
        self.node_id_to_coordinate_dict = node_id_to_coordinate_dict
        self.problem_type = problem_type
        self.program_options = program_options

    def add_and_check_tour(self, new_tour):
        if len(self.tour_history) is 0 or self.tour_history[-1] is not new_tour:
            self.tour_history.append(new_tour)

    def animate(self, output_directory_animation_graphs):
        """
        Animate the tour history that has been set. This uses the muiltiprocessing library to speed up the creation of
        graphs, runs os.cpu_count() number of processes at one time

        :param output_directory_animation_graphs:The directory where the temporary graphs should be saved
        """
        process_list = []
        file_name_list = []

        num = 0
        for tour in self.tour_history:
            file_name = output_directory_animation_graphs + str(num) + ".png"
            file_name_list.append(file_name)
            process = multiprocessing.Process(target=plot, args=(tour, file_name, self.node_id_to_coordinate_dict))
            process_list.append(process)
            num += 1

        start_num = 0
        process_amount = os.cpu_count()
        end_num = start_num + process_amount
        for _ in process_list:
            if start_num > len(process_list):
                break
            n_elements = process_list[start_num:end_num]
            for process in n_elements:
                process.start()

            for process in n_elements:
                process.join()

            start_num += process_amount
            end_num = min((end_num + process_amount), len(process_list))

        fps = max(int(len(file_name_list) / 10), 1)
        save_location = self.program_options.OUTPUT_DIRECTORY + self.program_options.TSP_PROBLEM_NAME + self.problem_type + ".mp4"
        with imageio.get_writer(save_location, mode='I', fps=fps) as writer:
            for file_name in file_name_list:
                image = imageio.imread(file_name)
                writer.append_data(image)

        writer.close()

        # for file_name in file_name_list:
            # os.remove(file_name)
