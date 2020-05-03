# From https://en.wikipedia.org/wiki/2-opt and
# https://www.researchgate.net/figure/Pseudo-code-for-2-opt-algorithm_tbl1_268981882
import logging

from plotting.tour_improvement_plotter import TourImprovementAnimator


def swap_2_opt(route: list, i, k):
    new_route = route[0:i]
    new_route.extend(reversed(route[i:k + 1]))
    new_route.extend(route[k + 1:])

    # Used to test that the swapping has been successful
    assert (len(new_route) == len(route))

    return new_route


def run_2_opt(existing_route, node_id_to_location_dict, distance_calculator_callback,
              tsp_2_opt_animator: TourImprovementAnimator):
    best_distance = distance_calculator_callback(existing_route, node_id_to_location_dict)
    best_route = existing_route

    improvement = True

    while improvement:
        improvement = False

        for i in range(len(best_route) - 1):
            for k in range(i + 1, len(best_route)):
                new_route = swap_2_opt(best_route, i, k)
                new_distance = distance_calculator_callback(new_route, node_id_to_location_dict)

                if new_distance < best_distance:
                    logging.debug("2-opt improvement found %s", new_distance)
                    tsp_2_opt_animator.add_and_check_tour(best_route)
                    best_route = new_route
                    best_distance = new_distance
                    improvement = True
                    break

    tsp_2_opt_animator.add_and_check_tour(best_route)
    return best_route
