from acopy import SolverPlugin

from plotting.tour_improvement_plotter import TourImprovementAnimator


class IterationPlotterPlugin(SolverPlugin):

    def __init__(self, tour_improvement_animator: TourImprovementAnimator, **kwargs):
        super().__init__(**kwargs)
        self.tour_improvement_animator = tour_improvement_animator

    def initialize(self, solver):
        super().initialize(solver)

    def on_iteration(self, state):
        if state.is_new_record:
            self.tour_improvement_animator.add_and_check_tour(state.record.nodes)
