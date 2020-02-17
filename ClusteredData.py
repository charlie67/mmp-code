class Cluster:
    nodes: list
    cluster_centre: tuple

    def __init__(self, cluster_centre, nodes) -> None:
        self.cluster_centre = cluster_centre
        self.nodes = nodes

    def add_nodes(self, node: list):
        self.nodes.extend(node)

    def get_nodes(self):
        return self.nodes

    def get_cluster_centre(self):
        return self.cluster_centre


class ClusteredData:
    # Class to hold the data to do with clustering to provide a common interface for the rest of the application
    # Holds the nodes and the clusters

    # Numpy array that holds all the nodes
    nodes: list

    # List of Cluster objects
    clusters: list

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
