perform_aco_over_clustered_problem(clustered_data.turn_clusters_into_nx_graph(problem))


def perform_aco_over_clustered_problem(graph):
    solver = acopy.Solver(rho=.03, q=1)
    colony = acopy.Colony(alpha=1, beta=10)
    printout_plugin = acopy.plugins.Printout()
    solver.add_plugin(printout_plugin)
    return solver.solve(graph, colony, limit=80)