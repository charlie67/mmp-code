from enum import Enum


class ClusterAlgorithmType(Enum):
    K_MEANS = "K-Means"
    AFFINITY_PROPAGATION = "Affinity Propagation"
    BIRCH = "Birch"
    DBSCAN = "DBSCAN"
    OPTICS = "Optics"
