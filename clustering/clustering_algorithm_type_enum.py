from enum import Enum


class ClusterAlgorithmType(Enum):
    K_MEANS = 1
    AFFINITY_PROPAGATION = 2
    BIRCH = 3
    DBSCAN = 4
    OPTICS = 5
