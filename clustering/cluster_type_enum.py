from enum import Enum


class ClusterType(Enum):
    FULL_CLUSTER = 1
    UNCLASSIFIED_NODE_CLUSTER = 2


class InternalClusterPathFinderType(Enum):
    ACO = 1
    GREEDY_NEAREST_NODE = 2
