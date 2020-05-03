import numpy as np


def calculate_distance_for_tour(tour, node_id_to_location_dict):
    length = 0
    num = 0

    for i in tour:
        j = tour[num - 1]
        distance = np.linalg.norm(node_id_to_location_dict[i] - node_id_to_location_dict[j])
        length += distance
        num += 1

    return length


def aco_distance_callback(node_1, node_2):
    x_distance = abs(node_1[0] - node_2[0])
    y_distance = abs(node_1[1] - node_2[1])

    # c = sqrt(a^2 + b^2)
    import math
    return math.sqrt(pow(x_distance, 2) + pow(y_distance, 2))
