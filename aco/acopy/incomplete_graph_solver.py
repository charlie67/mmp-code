# Came from https://github.com/rhgrant10/acopy/issues/21
import acopy


class IncompleteGraphSolver(acopy.Solver):
    """My custom solver for incomplete graphs.

    If the retry_limit is None then there is no limit. Careful when not setting
    this option since it means there could be infinite or otherwise time
    consuming loops.

    :param int retry_limit: maximum number of tries for each ant to find a tour
    """

    def __init__(self, retry_limit=None, rho=.03, q=1, end_node=None, start_node=None):
        super().__init__(rho=rho, q=q)
        self.end_node = end_node
        self.retry_limit = retry_limit
        self.start_node = start_node

    def _should_retry(self, tries):
        # return true if we should retry given the current number of tries
        if not self.retry_limit:
            return True
        return tries < self.retry_limit

    def find_solutions(self, graph, ants):
        """Return the solutions found for the given ants.

        :param graph: a graph
        :type graph: :class:`networkx.Graph`
        :param list ants: the ants to use
        :return: one solution per ant
        :rtype: list
        """
        # base implementation assumes every ant succeeds:
        # return [ant.tour(graph) for ant in ants]
        solutions = []
        tries = 0
        for ant in ants:
            # try until we have a solution or we hit the retry limit
            solution = None
            while solution is None and self._should_retry(tries):
                try:
                    solution = ant.tour(graph, end_node=self.end_node, start_node=self.start_node)
                except KeyError:
                    tries += 1  # try again
            # we may or may not have a solution for every ant
            if solution is not None:
                solutions.append(solution)
        return solutions
