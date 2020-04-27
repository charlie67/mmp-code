from aco.aco_type_enum import ACOType
from clustering.cluster_type_enum import InternalClusterPathFinderType
from clustering.clustering_algorithm_type_enum import ClusterAlgorithmType

NUMBER_CLUSTERS = 20
DISPLAY_PLOTS = False
ACO_TYPE = ACOType.ACO_PY
CLUSTER_TYPE = ClusterAlgorithmType.K_MEANS

AFFINITY_PROPAGATION_CONVERGENCE_ITERATIONS = 500
AFFINITY_PROPAGATION_MAX_ITERATIONS = 20000

OPTICS_MIN_SAMPLES = 5

K_MEANS_N_INIT = 10

BIRCH_BRANCHING_FACTOR = 50
BIRCH_THRESHOLD = 0.5

DBSCAN_EPS = 50
DBSCAN_MIN_SAMPLES = 3

# aco settings
ACO_ALPHA_VALUE = 1
ACO_BETA_VALUE = 3
ACO_RHO_VALUE = 0.03
ACO_Q_VALUE = 1
# If this is none then the defaults from the ACO libraries are used instead
ACO_ANT_COUNT = None
ACO_ITERATIONS = 100

SHOULD_RUN_2_OPT = True
SHOULD_CLUSTER = True

AUTOMATE_DBSCAN_EPS = False
PLT_DPI_VALUE = 800

CLUSTER_TOUR_TYPE = InternalClusterPathFinderType.ACO
