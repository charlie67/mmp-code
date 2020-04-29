import networkx as nx
import numpy as np

import acopy
from aco.acopy.incomplete_graph_solver import IncompleteGraphSolver
from aco.acopy.logger_plugin import LoggerPlugin
from aco.multithreaded.multi_threaded_ant_colony import AntColony
from clustering.cluster_type_enum import ClusterType
from distance_calculation import aco_distance_callback
from options.options_holder import Options
from plotting.graph_plotting import plot_tour_for_cluster


class Cluster:
    nodes: list
    cluster_centre: tuple

    # List of nodes, will contain two nodes when the movement between clusters has been calculated.
    # One will be the entry node and one is the exit node.
    # These are the nodes that are closest to the two neighbour nodes on the tour
    entry_exit_nodes: list

    # The tour for this cluster going from entry_exit_nodes[0] to entry_exit_nodes[1]
    tour: list

    cluster_type: ClusterType

    def __init__(self, cluster_centre, nodes, cluster_type: ClusterType, program_options: Options) -> None:
        self.program_options = program_options
        self.cluster_centre = cluster_centre
        self.nodes = nodes
        self.entry_exit_nodes = []
        self.cluster_type = cluster_type

    def add_nodes(self, node: list):
        self.nodes.extend(node)

    def get_nodes(self):
        return self.nodes

    def get_cluster_centre(self):
        return self.cluster_centre

    def turn_cluster_into_networkx_graph(self):
        nx_graph = nx.Graph()

        num = 0
        added_nodes = list()
        for i in self.nodes:
            if num is not self.entry_exit_nodes[1]:
                nx_graph.add_node(num, coord=i)
                added_nodes.append(num)
            num += 1

        for i in added_nodes:
            for j in added_nodes:
                if i == j:
                    continue
                distance = np.linalg.norm(self.nodes[i] - self.nodes[j])
                nx_graph.add_edge(i, j, weight=distance)

        return nx_graph

    def turn_cluster_into_node_id_to_location_dict(self):
        return_dict = dict()
        num = 0
        for node in self.nodes:
            # Only want one of the entry/exit nodes in the dict because the other gets added on at the end in order to
            # enforce the path because the aco library doesnt support setting the end node to a particular node
            if num is not self.entry_exit_nodes[1]:
                return_dict[num] = node
            num += 1

        return return_dict

    def calculate_tour_in_cluster_using_acopy(self):
        if len(self.nodes) is 2:
            answer = list()
            answer.append(self.entry_exit_nodes[0])
            answer.append(self.entry_exit_nodes[1])
            self.tour = answer
            return

        graph = self.turn_cluster_into_networkx_graph()

        solver = IncompleteGraphSolver(rho=self.program_options.ACO_RHO_VALUE, q=self.program_options.ACO_Q_VALUE,
                                       retry_limit=self.program_options.ACO_ITERATIONS,
                                       end_node=None,
                                       start_node=self.entry_exit_nodes[0])
        colony = acopy.Colony(alpha=self.program_options.ACO_ALPHA_VALUE, beta=self.program_options.ACO_BETA_VALUE)
        logger_plugin = LoggerPlugin()
        solver.add_plugin(logger_plugin)
        solution = solver.solve(graph, colony, limit=self.program_options.ACO_ITERATIONS,
                                gen_size=self.program_options.ACO_ANT_COUNT)
        answer = solution.nodes

        # Need to ensure answer starts with entry_exit_nodes[0]
        index_first_entry_exit_node = answer.index(self.entry_exit_nodes[0])
        if index_first_entry_exit_node is not 0:
            new_route = answer[index_first_entry_exit_node:]
            new_route.extend(answer[0:index_first_entry_exit_node])
            answer = new_route

        # Put the cluster exit node that isn't present in the ACO on the end
        answer.append(self.entry_exit_nodes[1])

        self.tour = answer

    def calculate_tour_in_cluster_using_muiltithreaded_aco(self):
        cluster_nodes_dict = self.turn_cluster_into_node_id_to_location_dict()
        colony = AntColony(nodes=cluster_nodes_dict, distance_callback=aco_distance_callback,
                           alpha=self.program_options.ACO_ALPHA_VALUE,
                           beta=self.program_options.ACO_BETA_VALUE,
                           pheromone_evaporation_coefficient=self.program_options.ACO_RHO_VALUE,
                           pheromone_constant=self.program_options.ACO_Q_VALUE,
                           ant_count=self.program_options.ACO_ANT_COUNT, iterations=self.program_options.ACO_ITERATIONS,
                           start=self.entry_exit_nodes[0])
        answer = colony.mainloop()
        answer.append(self.entry_exit_nodes[1])
        self.tour = answer

    def calculate_tour_in_cluster_using_closest_node(self):
        # start at start node and go to the closest node that hasn't been visited

        tour = []
        graph = self.turn_cluster_into_networkx_graph()
        start_node = self.entry_exit_nodes[0]
        end_node = self.entry_exit_nodes[1]
        tour.append(start_node)

        # go from the start node and find the closest node that is not in the tour, when all the nodes have been visited
        # move to the exit node and complete
        for i in range(len(self.nodes) - 1):
            # check if this is the last iteration and if so just add on the end node
            if len(tour) is len(self.nodes) - 1:
                tour.append(end_node)
                break

            # find the closest node to tour[i] that isn't already in the tour or end_node
            node_number = tour[i]
            nodes = graph.neighbors(node_number)
            closest_node = None
            closest_node_distance = None

            for node in nodes:
                edge_information = graph.get_edge_data(node_number, node)
                distance = edge_information['weight']

                if (
                        closest_node_distance is None or distance < closest_node_distance) and node not in tour and node is not end_node:
                    closest_node = node
                    closest_node_distance = distance

            tour.append(closest_node)

        self.tour = tour
        return tour

    def return_tour_ordered_list(self):
        tour_ordered_nodes = []
        # If this cluster is a full cluster containing multiple nodes then it will have a tour so go over that tour
        # and create an ordered list. If this cluster is for a node that couldn't be placed into a tour then just
        # add the only node to the list and return that
        if self.cluster_type is ClusterType.FULL_CLUSTER and len(self.nodes) > 1:
            for node in self.tour:
                tour_ordered_nodes.append(self.nodes[node])

            return tour_ordered_nodes

        tour_ordered_nodes.append(self.nodes[0])
        return tour_ordered_nodes

    def plot_graph_of_tour(self, cluster_title):
        plot_tour_for_cluster(self.tour, self.nodes, self.program_options, cluster_title, self.entry_exit_nodes)


def get_cluster_centres(clusters):
    cluster_centres = np.zeros(shape=(len(clusters), 2))
    i = 0

    for c in clusters:
        cluster_centres[i, 0] = c.get_cluster_centre()[0]
        cluster_centres[i, 1] = c.get_cluster_centre()[1]
        i += 1

    return cluster_centres


def move_between_two_clusters(cluster_a: Cluster, cluster_b: Cluster):
    """
    Find the shortest path between two given clusters. Will find the node in cluster a that is closest to the centre of b
    and then find the node in b that is closest to this node in a
    Will fill out the clusters with the relevant entry/exit nodes
    move from cluster_a to cluster_b
    get the centroid of cluster b
    and then find the node in cluster_a that is closest to this centroid
    then find the node in cluster_b that is closest to this new closest cluster_a value
    """

    closest_cluster_a = None
    closest_cluster_a_distance = None
    cluster_a_node_number = None

    centre = cluster_b.get_cluster_centre()
    counter = 0
    for node in cluster_a.get_nodes():
        # calculate the distance between node and centre
        distance = np.linalg.norm(node - centre)

        if (
                closest_cluster_a_distance is None or distance < closest_cluster_a_distance or closest_cluster_a is None) and (
                counter not in cluster_a.entry_exit_nodes or len(cluster_a.nodes) == 1):
            closest_cluster_a_distance = distance
            closest_cluster_a = node
            cluster_a_node_number = counter
        counter += 1

    closest_cluster_b = None
    closest_cluster_b_distance = None
    cluster_b_node_number = None

    counter = 0
    for node in cluster_b.get_nodes():
        # calculate the distance between node and centre
        distance = np.linalg.norm(node - closest_cluster_a)

        if (
                closest_cluster_b_distance is None or distance < closest_cluster_b_distance or closest_cluster_b is None) and (
                counter not in cluster_b.entry_exit_nodes or len(cluster_b.nodes) == 1):
            closest_cluster_b_distance = distance
            closest_cluster_b = node
            cluster_b_node_number = counter
        counter += 1

    cluster_a.entry_exit_nodes.append(cluster_a_node_number)
    cluster_b.entry_exit_nodes.append(cluster_b_node_number)


def move_between_clusters_as_two_closest_nodes(cluster_a: Cluster, cluster_b: Cluster):
    closest_distance = None

    cluster_a_node_number = None
    cluster_b_node_number = None

    counter_a = 0

    for node_a in cluster_a.get_nodes():
        counter_b = 0
        for node_b in cluster_b.get_nodes():
            # calculate the distance between the two nodes
            distance = np.linalg.norm(node_a - node_b)

            if (closest_distance is None or distance < closest_distance) and (
                    counter_a not in cluster_a.entry_exit_nodes or len(cluster_a.nodes) == 1) and (
                    counter_b not in cluster_b.entry_exit_nodes or len(cluster_b.nodes) == 1):
                closest_distance = distance
                cluster_a_node_number = counter_a
                cluster_b_node_number = counter_b

            counter_b += 1

        counter_a += 1

    cluster_a.entry_exit_nodes.append(cluster_a_node_number)
    cluster_b.entry_exit_nodes.append(cluster_b_node_number)


class ClusteredData:
    # Class to hold the data to do with clustering to provide a common interface for the rest of the application
    # Holds the nodes and the clusters

    # Numpy array that holds all the nodes
    nodes: list

    # List of Cluster objects, these are all full cluster objects and contain multiple nodes
    clusters: list

    # list of nodes that couldn't be clustered, these are all cluster objects.
    # These are just single nodes in the form of a cluster object
    unclassified_nodes: list

    # The tour for only the clustered and unclusterable data
    # This is the tour that came from the ACO run
    aco_cluster_tour: list

    # The tour for all the nodes
    # This is the ACO tour but expanded into the clusters
    node_level_tour: list

    # Dict where the key is the node location and the value is the node id
    node_location_to_id_dict: dict

    # Dict where the key is the node id and the value is the location
    node_id_to_location_dict: dict

    def __init__(self, nodes, clusters, program_options: Options):
        self.program_options = program_options
        self.nodes = nodes
        self.clusters = clusters
        self.unclassified_nodes = []

    def get_clusters(self):
        return self.clusters

    def add_cluster(self, cluster: Cluster):
        self.clusters.append(cluster)

    def add_unclassified_node(self, node):
        self.unclassified_nodes.append(node)

    def get_unclassified_nodes(self):

        return self.unclassified_nodes

    def get_dict_node_id_location_mapping_aco(self):
        return_dict = dict()

        num = 0
        for node in self.clusters:
            return_dict[num] = node.cluster_centre
            num += 1

        for node in self.unclassified_nodes:
            return_dict[num] = node.cluster_centre
            num += 1

        return return_dict

    def get_all_clusters(self):
        """Clusters are stored in two different lists, the clusters that came from nodes that couldn't be classified are
        separated from the algorithm derived clusters that contain multiple nodes.
        This method combines both of those lists (if appropriate) and returns that
        :return: A list containing all the clusters including the unclassified nodes (if they exist)
        """

        if len(self.unclassified_nodes) > 0:
            list_to_return = []

            list_to_return.extend(self.clusters)
            list_to_return.extend(self.unclassified_nodes)
            return list_to_return

        return self.clusters

    # Get and return the centres of all the clusters
    def get_all_cluster_centres_and_unclassified_node_locations(self):
        if len(self.get_unclassified_nodes()) > 0:
            return np.append(get_cluster_centres(self.clusters), get_cluster_centres(self.unclassified_nodes), axis=0)
        return get_cluster_centres(self.clusters)

    # Go over the ACO tour and find a path connecting each cluster
    def find_nodes_to_move_between_clusters(self):
        c = 0
        nodes_in_tour = self.get_all_clusters()

        for node in self.aco_cluster_tour:
            j = self.aco_cluster_tour[c - 1]
            move_between_clusters_as_two_closest_nodes(nodes_in_tour[j], nodes_in_tour[node])

            c += 1

    def turn_clusters_into_nx_graph(self, tsplib_problem):
        """
        Turn all the clusters and unclassified nodes into a network x graph.
        :param tsplib_problem: The TSP problem as a TSPLIB 95 problem file
        :return: A network x graph containing the centre points of all the clusters (including unclassified nodes)
        """
        cluster_centres = get_cluster_centres(self.clusters)
        nx_graph = nx.Graph() if tsplib_problem.is_symmetric() else nx.DiGraph()
        nx_graph.graph['name'] = tsplib_problem.name
        nx_graph.graph['comment'] = tsplib_problem.comment
        nx_graph.graph['type'] = tsplib_problem.type
        nx_graph.graph['dimension'] = tsplib_problem.dimension
        nx_graph.graph['capacity'] = tsplib_problem.capacity
        nx_graph.graph['depots'] = tsplib_problem.depots
        nx_graph.graph['demands'] = tsplib_problem.demands
        nx_graph.graph['fixed_edges'] = tsplib_problem.fixed_edges
        num = 0
        for i in cluster_centres:
            nx_graph.add_node(num, coord=i)
            num += 1

        if len(self.get_unclassified_nodes()) > 0:
            for i in self.get_unclassified_nodes():
                nx_graph.add_node(num, coord=i)
                num += 1

        all_nodes = self.get_all_cluster_centres_and_unclassified_node_locations()
        for i in range(num):
            for j in range(num):
                if i == j:
                    continue
                distance = np.linalg.norm(all_nodes[i] - all_nodes[j])
                nx_graph.add_edge(i, j, weight=distance)

        return nx_graph

    def plot_all_cluster_tours(self):
        num = 0

        for cluster in self.clusters:
            if cluster.cluster_type is ClusterType.FULL_CLUSTER:
                cluster.plot_graph_of_tour("cluster" + str(num))
                num += 1

    def find_tours_within_clusters_using_greedy_closest_nodes(self):
        for cluster in self.clusters:
            if cluster.cluster_type is ClusterType.FULL_CLUSTER:
                cluster.calculate_tour_in_cluster_using_closest_node()

    def find_tours_within_clusters_using_acopy(self):
        for cluster in self.clusters:
            if cluster.cluster_type is ClusterType.FULL_CLUSTER and len(cluster.nodes) > 1:
                cluster.calculate_tour_in_cluster_using_acopy()

    def find_tours_within_clusters_using_multithreaded_aco(self):
        for cluster in self.clusters:
            if cluster.cluster_type is ClusterType.FULL_CLUSTER and len(cluster.nodes) > 1:
                cluster.calculate_tour_in_cluster_using_muiltithreaded_aco()

    def get_ordered_nodes_for_all_clusters(self):
        ordered_nodes = []
        all_clusters = self.get_all_clusters()

        for node in self.aco_cluster_tour:
            cluster = all_clusters[node]
            ordered_nodes.extend(cluster.return_tour_ordered_list())

        return ordered_nodes
