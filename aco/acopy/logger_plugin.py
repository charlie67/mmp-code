from acopy import SolverPlugin
import logging


class LoggerPlugin(SolverPlugin):
    """ Based on the printout plugin but uses a logger instead so that it gets printed out to the file and saved
    """
    iteration: int

    _ROW = '{:<10} {:<20} {}'

    def initialize(self, solver):
        super().initialize(solver)
        self.iteration = 0

    def on_start(self, state):
        self.iteration = 0
        logging.debug(f'Using {state.gen_size} ants from {state.colony}')
        logging.debug(f'Performing {state.limit} iterations:')
        logging.debug(self._ROW.format('Iteration', 'Cost', 'Solution'))

    def on_iteration(self, state):
        self.iteration += 1
        line = self._ROW.format(self.iteration, state.best.cost,
                                state.best.get_easy_id())

        if state.is_new_record:
            logging.debug(line)

    def on_finish(self, state):
        logging.debug('Done' + ' ' * (32 + 2 * len(state.graph)))