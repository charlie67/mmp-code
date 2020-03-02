import networkx as nx
import numpy as np


class Cluster:
    nodes: list
    cluster_centre: tuple

    # List of nodes, will contain two nodes when the movement between clusters has been calculated.
    # One will be the entry node and one is the exit node.
    # These are the nodes that are closest to the two neighbour nodes on the tour
    entry_exit_nodes: list

    def __init__(self, cluster_centre, nodes) -> None:
        self.cluster_centre = cluster_centre
        self.nodes = nodes
        self.entry_exit_nodes = []

    def add_nodes(self, node: list):
        self.nodes.extend(node)

    def get_nodes(self):
        return self.nodes

    def get_cluster_centre(self):
        return self.cluster_centre


def get_cluster_centres(clusters):
    cluster_centres = np.zeros(shape=(len(clusters), 2))
    i = 0

    for c in clusters:
        cluster_centres[i, 0] = c.get_cluster_centre()[0]
        cluster_centres[i, 1] = c.get_cluster_centre()[1]
        i += 1

    return cluster_centres


class ClusteredData:
    # Class to hold the data to do with clustering to provide a common interface for the rest of the application
    # Holds the nodes and the clusters

    # Numpy array that holds all the nodes
    nodes: list

    # List of Cluster objects
    clusters: list

    # list of nodes that couldn't be clustered
    unclassified_nodes: list = []

    def __init__(self, nodes, clusters):
        self.nodes = nodes
        self.clusters = clusters

    # Get all the nodes for this dataset
    def get_nodes(self) -> list:
        return self.nodes

    def get_clusters(self):
        return self.clusters

    def add_cluster(self, cluster: Cluster):
        self.clusters.append(cluster)

    def add_unclassified_node(self, node):
        self.unclassified_nodes.append(node)

    def get_unclassified_nodes(self):
        return self.unclassified_nodes

    def get_all_overall_nodes_as_clusters(self):
        if len(self.get_unclassified_nodes()) > 0:
            list_to_return = []

            list_to_return.extend(self.clusters)
            list_to_return.extend(self.unclassified_nodes)
            return list_to_return

        return self.clusters

    def get_all_overall_nodes(self):
        if len(self.get_unclassified_nodes()) > 0:
            return np.append(get_cluster_centres(self.clusters), get_cluster_centres(self.unclassified_nodes), axis=0)
        return get_cluster_centres(self.clusters)

    def turn_clusters_into_nx_graph(self, tsplib_problem):
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

        all_nodes = self.get_all_overall_nodes()
        for i in range(num):
            for j in range(num):
                if i == j:
                    continue
                distance = np.linalg.norm(all_nodes[i] - all_nodes[j])
                nx_graph.add_edge(i, j, weight=distance)

        return nx_graph
