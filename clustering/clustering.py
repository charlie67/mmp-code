import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AffinityPropagation, KMeans, Birch, DBSCAN, OPTICS

from clustering import dbscan_eps_finder
from clustering.cluster_type_enum import ClusterType
from clustering.clustered_data import ClusteredData, Cluster
from options.options_holder import Options


def perform_affinity_propagation(data, program_options: Options) -> ClusteredData:
    # The data that will be returned
    clustered_data = ClusteredData(data, list(), program_options=program_options)

    af = AffinityPropagation(convergence_iter=program_options.AFFINITY_PROPAGATION_CONVERGENCE_ITERATIONS,
                             max_iter=program_options.AFFINITY_PROPAGATION_MAX_ITERATIONS).fit(data)
    affinity_propagation_cluster_centers_indices = af.cluster_centers_indices_
    affinity_propagation_labels = af.labels_
    n_clusters_ = len(affinity_propagation_cluster_centers_indices)
    print('Estimated number of AffinityPropagation clusters: %d' % n_clusters_)

    for k in range(n_clusters_):
        class_members = affinity_propagation_labels == k
        cluster_center = data[affinity_propagation_cluster_centers_indices[k]]

        cluster = Cluster(cluster_centre=cluster_center, nodes=data[class_members],
                          cluster_type=ClusterType.FULL_CLUSTER, program_options=program_options)
        clustered_data.add_cluster(cluster)

    return clustered_data


def perform_optics_clustering(data, program_options: Options) -> ClusteredData:
    # The data that will be returned
    clustered_data = ClusteredData(data, list(), program_options=program_options)

    op = OPTICS(min_samples=program_options.OPTICS_MIN_SAMPLES, n_jobs=-1)
    op.fit(data)
    optic_labels = op.labels_

    for k in range(optic_labels.max() + 1):
        class_members = optic_labels == k
        nodes_in_cluster = data[class_members]
        # optics has no way of telling you the final cluster centres so have to calculate it yourself
        cluster_centre = nodes_in_cluster.mean(axis=0)
        cluster = Cluster(cluster_centre=cluster_centre, nodes=nodes_in_cluster, cluster_type=ClusterType.FULL_CLUSTER,
                          program_options=program_options)
        clustered_data.add_cluster(cluster)

    if optic_labels.min() == -1:
        class_members = optic_labels == -1
        # There are unclassified nodes
        unclassified_nodes = data[class_members]
        for unclassified_node in unclassified_nodes:
            cluster_to_add = Cluster(unclassified_node, [unclassified_node],
                                     cluster_type=ClusterType.UNCLASSIFIED_NODE_CLUSTER,
                                     program_options=program_options)
            clustered_data.add_unclassified_node(cluster_to_add)

    return clustered_data


def perform_k_means_clustering(data, program_options: Options) -> ClusteredData:
    # The data that will be returned
    clustered_data = ClusteredData(data, list(), program_options=program_options)

    km = KMeans(init='k-means++', n_clusters=program_options.NUMBER_CLUSTERS, n_init=program_options.K_MEANS_N_INIT,
                n_jobs=-1)
    km.fit(data)
    k_mean_labels = km.predict(data)
    k_means_cluster_centers_indices = km.cluster_centers_
    n_clusters_ = len(k_means_cluster_centers_indices)
    for k in range(n_clusters_):
        class_members = k_mean_labels == k
        cluster = Cluster(cluster_centre=k_means_cluster_centers_indices[k], nodes=data[class_members],
                          cluster_type=ClusterType.FULL_CLUSTER, program_options=program_options)
        clustered_data.add_cluster(cluster)

    print("k-mean clusters", k_mean_labels)
    return clustered_data


def perform_birch_clustering(data, program_options: Options) -> ClusteredData:
    # The data that will be returned
    clustered_data = ClusteredData(data, list(), program_options=program_options)

    brc = Birch(branching_factor=program_options.BIRCH_BRANCHING_FACTOR, n_clusters=program_options.NUMBER_CLUSTERS,
                threshold=program_options.BIRCH_THRESHOLD)
    brc.fit(data)
    birch_labels = brc.predict(data)

    for k in range(brc.n_clusters):
        class_members = birch_labels == k
        nodes_in_cluster = data[class_members]
        # birch has no way of telling you the final cluster centres so have to calculate it yourself
        cluster_centre = nodes_in_cluster.mean(axis=0)
        cluster = Cluster(cluster_centre=cluster_centre, nodes=nodes_in_cluster, cluster_type=ClusterType.FULL_CLUSTER,
                          program_options=program_options)
        clustered_data.add_cluster(cluster)

    print("birch clusters", birch_labels)

    return clustered_data


def perform_dbscan_clustering(data, program_options: Options) -> ClusteredData:
    if program_options.AUTOMATE_DBSCAN_EPS:
        program_options.DBSCAN_EPS = dbscan_eps_finder.find_using_nearest_neighbours(problem_data_array=data,
                                                                                     program_options=program_options)

    # The data that will be returned
    clustered_data = ClusteredData(data, list(), program_options)

    db = DBSCAN(eps=program_options.DBSCAN_EPS, min_samples=program_options.DBSCAN_MIN_SAMPLES, n_jobs=-1).fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    db_labels = db.labels_
    db_n_clusters_ = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise_ = list(db_labels).count(-1)

    for k in range(db_n_clusters_):
        class_members = db_labels == k
        nodes_in_cluster = data[class_members]
        cluster_centre = nodes_in_cluster.mean(axis=0)
        cluster = Cluster(cluster_centre=cluster_centre, nodes=nodes_in_cluster, cluster_type=ClusterType.FULL_CLUSTER,
                          program_options=program_options)
        clustered_data.add_cluster(cluster)

    # These are the nodes that could not be placed into a cluster
    if n_noise_ > 0:
        class_members = db_labels == -1
        unclassified_nodes = data[class_members]
        for unclassified_node in unclassified_nodes:
            cluster_to_add = Cluster(unclassified_node, [unclassified_node],
                                     cluster_type=ClusterType.UNCLASSIFIED_NODE_CLUSTER,
                                     program_options=program_options)
            clustered_data.add_unclassified_node(cluster_to_add)

    return clustered_data


# Plot a graph that contains every cluster and unclustered node. The clusters will have edges connecting themselves
# to the nodes in their cluster.
def plot_clustered_graph(plot_colours, cluster_data: ClusteredData, program_options):
    # This plotting was adapted from the affinity propagation sklearn example
    i = 0
    for k, col in zip(range(len(cluster_data.get_clusters())), plot_colours):
        class_members = cluster_data.get_clusters()[k].nodes
        cluster_center = cluster_data.get_clusters()[k].get_cluster_centre()

        plt.plot(class_members[:, 0], class_members[:, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
        plt.annotate(i, xy=(cluster_center[0], cluster_center[1]), fontsize=10, ha='center', va='center')
        i += 1

        for x in class_members:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col, linewidth=0.5)

    unclassified_nodes = cluster_data.get_unclassified_nodes()
    if len(unclassified_nodes) > 0:
        for k in unclassified_nodes:
            plt.plot(k.cluster_centre[0], k.cluster_centre[1], 'o', markerfacecolor='k', markeredgecolor='k',
                     markersize=6)

    plt.title(program_options.TSP_PROBLEM_NAME + ' ' + str(program_options.CLUSTER_TYPE) + ': clusters: %d noise: %d' % (
    len(cluster_data.get_clusters()), len(cluster_data.get_unclassified_nodes())))
    plt.savefig(program_options.OUTPUT_DIRECTORY + program_options.TSP_PROBLEM_NAME + "-" + str(program_options.CLUSTER_TYPE) + "-clustering.png", dpi=program_options.PLT_DPI_VALUE)

    if program_options.DISPLAY_PLOTS:
        plt.show()

    plt.close()
