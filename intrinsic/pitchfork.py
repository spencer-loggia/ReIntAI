import torch


class Pitchfork():
    """
    returns instances that have been optimized by a :
    - Gradient based optimizer (Adam)
    - Genetic Algorithm
    Maintains states and meta info related to optimization trajectories.
    """
    def __init__(self, params, meta_params: dict, lr=0.01):
        super().__init__()
        self.params = list(params)
        self.meta_params = meta_params
        self.adam = torch.optim.Adam(params, lr=lr)
        self.snapshot = None
        self.adam_snapshot = None
        self.last_score = None

    def step(self):
        self.snapshot = self.params.copy()
        self.adam_snapshot = self.adam.state_dict()
        self.adam.step()




