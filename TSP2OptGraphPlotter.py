import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation

import time


class TSP2OptAnimator:
    graph_file_names: list

    tour_history: list

    node_id_to_coordinate_dict: dict

    def __init__(self, node_id_to_coordinate_dict) -> None:
        self.graph_file_names = list()
        self.tour_history = list()
        self.node_id_to_coordinate_dict = node_id_to_coordinate_dict

    def plot(self, tour):
        num = 0
        figure = plt.figure(figsize=[40, 40])
        for i in tour:
            j = tour[num - 1]

            node_i = self.node_id_to_coordinate_dict[i]
            node_j = self.node_id_to_coordinate_dict[j]

            plt.plot(node_i[0], node_i[1], 'b.', markersize=10, figure=figure)
            plt.plot([node_i[0], node_j[0]], [node_i[1], node_j[1]], 'k', linewidth=0.5, figure=figure)
            num += 1

        file_name = "/tmp/tour" + str(time.time()) + ".png"
        plt.savefig(file_name)
        self.graph_file_names.append(file_name)
        plt.clf()

    def add_and_check_tour(self, new_tour):
        if len(self.tour_history) is 0:
            self.tour_history.append(new_tour)

        last_tour = self.tour_history[-1]

        if new_tour is not last_tour:
            self.tour_history.append(new_tour)

    def animate(self):
        for tour in self.tour_history:
            self.plot(tour)

        frames = []
        images = []
        fig = plt.figure(figsize=[20, 20])

        for file_name in self.graph_file_names:
            images.append(mpimg.imread(file_name))

        for i in images:
            frames.append([plt.imshow(i, animated=True)])

        ani = animation.ArtistAnimation(fig, frames, interval=2000, blit=True,
                                        repeat_delay=1000)
        ani.save('movie.mp4')

        for file_name in self.graph_file_names:
            os.remove(file_name)
