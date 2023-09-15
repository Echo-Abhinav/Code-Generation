class DecodeHypothesis(object):
    def __init__(self):
        self.actions = []
        self.score = 0.
        self.rules = [(None, '-')]
        self.stack = []
        self.pointer = []
        # record the current time step
        self.t = 0

    def copy(self):
        new_hyp = DecodeHypothesis()

        new_hyp.actions = list(self.actions).copy()
        new_hyp.score = self.score
        new_hyp.rules = list(self.rules).copy()
        new_hyp.t = self.t
        new_hyp.stack = self.stack.copy()
        new_hyp.pointer = self.pointer.copy()

        return new_hyp

