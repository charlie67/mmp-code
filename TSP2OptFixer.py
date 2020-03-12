# From https://en.wikipedia.org/wiki/2-opt and https://github.com/rellermeyer/99tsp/blob/master/python/2opt/TSP2opt.py
import numpy as np
import matplotlib.pyplot as plt


def swap_2_opt(route: list, i, k):
    new_route = route[0:i]
    new_route.extend(reversed(route[i:k + 1]))
    new_route.extend(route[k+1:])

    assert(len(new_route) == len(route))

    return new_route


def run_2_opt(existing_route, node_id_to_location_dict, calculate_distance):
    best_distance = calculate_distance(existing_route, node_id_to_location_dict)
    best_route = existing_route

    improvement = True

    while improvement:
        improvement = False

        for i in range(len(best_route) - 1):
            for k in range(i + 1, len(best_route)):
                new_route = swap_2_opt(best_route, i, k)
                new_distance = calculate_distance(new_route, node_id_to_location_dict)

                if new_distance < best_distance:
                    plot_complete_tsp_tour(best_route, node_id_to_location_dict)
                    best_route = new_route
                    best_distance = new_distance
                    improvement = True
                    break

        if improvement:
            break

    return best_route


def plot_complete_tsp_tour(tour, node_id_to_coordinate_dict):
    num = 0
    figure = plt.figure(figsize=[40, 40])
    for i in tour:
        j = tour[num - 1]

        node_i = node_id_to_coordinate_dict[i]
        node_j = node_id_to_coordinate_dict[j]

        plt.plot(node_i[0], node_i[1], 'b.', markersize=10, figure=figure)
        plt.plot([node_i[0], node_j[0]], [node_i[1], node_j[1]], 'k', linewidth=0.5, figure=figure)
        num += 1
    plt.show()
