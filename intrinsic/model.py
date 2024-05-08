import torch
import numpy as np
from intrinsic.module import PlasticEdges, FCPlasticEdges


class Intrinsic:
    """
    Parallelized model.
    """

    def __init__(self, num_nodes, node_shape: tuple = (1, 3, 64, 64), inject_noise=False,
                 edge_module=PlasticEdges, device='cpu', track_activation_history=False,
                 mask=None, kernel_size=3, is_resistive=True, input_mode="overwrite",
                 optimize_weights=True):
        """
        :param num_nodes: Number of nodes in the graph.
        :param node_shape: Shape (channels and spatial of each node in the graph.
        :param inject_noise: Whether to inject random noise at each step.
        :param edge_module: Class to use for edges.
        :param device: Hardware device to use for computation.
        :param input_mode: How inputs should be injected into the graph.
        :param track_activation_history: Whether to store the state history
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.resistive = is_resistive
        if input_mode not in ["additive", "overwrite"]:
            raise ValueError
        self.input_mode = input_mode

        # how much of previous state to mix into next input.
        # TODO: Perhaps there should be a "resting" state value that is exponentially returned to with time constant
        #  resistance? (tried and seems to make optimization less stable, mb worth revisiting.)
        self.resistance = torch.nn.Parameter(torch.zeros((num_nodes, node_shape[1], 1, 1), device=device))
        # Initialize (n, c, s, s) state matrix using xavier method.
        states = torch.empty(size=(self.num_nodes, node_shape[1], node_shape[2], node_shape[3]), device=device)
        self.states = torch.nn.init.xavier_normal_(states)
        # (n,n) adjacent matrix mask sets immutable modifier on each edge.
        if mask is None:
            mask = torch.ones((num_nodes, num_nodes), device=device)

        self.edge_module = edge_module
        # edge module takes n x c x s1 x s2 input and returns output of the same shape.
        self.edge = edge_module(self.num_nodes, node_shape[2], node_shape[3], kernel_size=kernel_size,
                                channels=node_shape[1],
                                device=device, mask=mask, inject_noise=inject_noise, normalize_conv=False,
                                init_plasticity=.2,
                                optimize_weights=optimize_weights)

        # whether to add random gaussian noise at each forward step
        self.inject_noise = inject_noise
        self.noise = .1  # std deviations of injected noise.
        self.sigmoid = torch.nn.Sigmoid()  # activation function for each edge must have range on closed interval [0, 1]
        # memory to track history of each state.
        if track_activation_history:
            self.past_states = []
        else:
            self.past_states = None
        # hardware device to run stuff on.
        self.device = device

    def instantiate(self):
        new_model = Intrinsic(self.num_nodes, (1, self.edge.channels, self.edge.spatial1, self.edge.spatial2),
                              inject_noise=self.inject_noise, edge_module=PlasticEdges, device=self.device,
                              track_activation_history=self.past_states is not None, mask=self.edge.mask,
                              kernel_size=self.edge.kernel_size, is_resistive=self.resistive,
                              input_mode=self.input_mode,
                              optimize_weights=self.edge.optimize_weights)
        new_model.states = self.states
        new_model.edge = self.edge.instantiate()
        new_model.resistance = self.resistance.clone()
        return new_model

    def __call__(self, x=None, mask=None):
        return self.forward(x, mask)

    def forward(self, x=None, mask=None):
        """
        :param x: optional. Tensor with the same shape as the states.
        :param mask: boolean, optional, required with x. Which state indexes are updatable by x
        :return:
        """
        h = self.states - 1  # + torch.normal(0, self.noise, self.states.shape)  # inject noise (and subtract 1?)
        self.edge.update(h)  # do local weight update
        out_activ = self.edge(h).clone()  # get output from all edges.

        if x is not None:
            if x.shape == self.states.shape:
                if self.input_mode == "overwrite":
                    # used state values should be != 0.
                    out_activ = (out_activ * torch.logical_not(mask)) + x
                elif self.input_mode == "additive":
                    out_activ = out_activ + x
            else:
                raise IndexError
        # mix state update and current state values parameterized by resistance.
        self.states = self.states * self.resistance + out_activ
        if self.past_states is not None:
            self.past_states.append(self.states.clone())
        return self.states

    def detach(self, reset_intrinsic=False):
        # detach computational graph
        self.edge.detach(reset_weight=reset_intrinsic)
        if reset_intrinsic:
            states = torch.zeros_like(self.states)
            self.states = torch.nn.init.xavier_normal_(states)
        else:
            self.states = self.states.detach().clone()
        self.past_states = []
        return self

    def parameters(self, recurse: bool = True):
        """
        :return: list of parameters that can be optimized by gradient descent.
        """
        params = self.edge.parameters() + [self.resistance]
        return params

    def set_grad(self, grads):
        self.resistance.grad = grads[-1]
        self.edge.set_grad(grads[:-1])

    def l1(self):
        ps = self.parameters()
        penalty = torch.sum(torch.stack([torch.sum(torch.abs(p)) for p in ps]))
        return penalty

    def clone(self, fuzzy=False, device=None):
        if device is None:
            device = self.device
        new_model = Intrinsic(self.num_nodes, (1, self.edge.channels, self.edge.spatial1, self.edge.spatial2),
                              inject_noise=self.inject_noise, edge_module=PlasticEdges, device=device,
                              track_activation_history=self.past_states is not None, mask=self.edge.mask,
                              kernel_size=self.edge.kernel_size, is_resistive=self.resistive,
                              input_mode=self.input_mode,
                              optimize_weights=self.edge.optimize_weights)
        new_model.states = self.states.detach().to(device)
        new_model.edge = self.edge.clone(fuzzy=fuzzy).to(device)
        new_model.resistance = torch.nn.Parameter(self.resistance.detach().clone().to(device))
        return new_model

    def to(self, device):
        self.device = device
        return self.clone(fuzzy=False, device=device)


class FCIntrinsic:
    """
      Parallelized model.
      """

    def __init__(self, num_nodes, node_shape: tuple = (1, 3, 64), inject_noise=False,
                 edge_module=FCPlasticEdges, device='cpu', track_activation_history=False,
                 mask=None, is_resistive=True, input_mode="overwrite",
                 optimize_weights=True, *args, **kwargs):
        """
        :param num_nodes: Number of nodes in the graph.
        :param node_shape: Shape (channels and spatial of each node in the graph.
        :param inject_noise: Whether to inject random noise at each step.
        :param edge_module: Class to use for edges.
        :param device: Hardware device to use for computation.
        :param input_mode: How inputs should be injected into the graph.
        :param track_activation_history: Whether to store the state history
        """
        self.resistive = is_resistive
        if input_mode not in ["additive", "overwrite"]:
            raise ValueError
        self.input_mode = input_mode

        # how much of previous state to mix into next input.
        self.kernel_size = None
        self.num_nodes = num_nodes
        self.resistance = torch.nn.Parameter(torch.zeros((num_nodes, node_shape[1], 1), device=device))
        # Initialize (n, c, s, s) state matrix using xavier method.
        states = torch.empty(size=(self.num_nodes, node_shape[1], node_shape[2]), device=device)
        self.states = torch.nn.init.xavier_normal_(states)
        # (n,n) adjacent matrix mask sets immutable modifier on each edge.
        if mask is None:
            mask = torch.ones((num_nodes, num_nodes), device=device)

        self.edge_module = edge_module
        # edge module takes n x c x s1 x s2 input and returns output of the same shape.
        self.edge = edge_module(self.num_nodes, node_shape[2],
                                channels=node_shape[1],
                                device=device, mask=mask, inject_noise=inject_noise, normalize_conv=False,
                                init_plasticity=.2,
                                optimize_weights=optimize_weights)

        # whether to add random gaussian noise at each forward step
        self.inject_noise = inject_noise
        self.noise = .1  # std deviations of injected noise.
        self.sigmoid = torch.nn.Sigmoid()  # activation function for each edge must have range on closed interval [0, 1]
        # memory to track history of each state.
        if track_activation_history:
            self.past_states = []
        else:
            self.past_states = None
        # hardware device to run stuff on.
        self.device = device

    def instantiate(self):
        new_model = FCIntrinsic(self.num_nodes, (1, self.edge.channels, self.edge.spatial),
                                inject_noise=self.inject_noise, edge_module=FCPlasticEdges, device=self.device,
                                track_activation_history=self.past_states is not None, mask=self.edge.mask,
                                is_resistive=self.resistive,
                                input_mode=self.input_mode,
                                optimize_weights=self.edge.optimize_weights)
        new_model.states = self.states
        new_model.edge = self.edge.instantiate()
        new_model.resistance = self.resistance.clone()
        return new_model

    def __call__(self, x=None, mask=None):
        return self.forward(x, mask)

    def forward(self, x=None, mask=None):
        """
        :param x: optional. Tensor with the same shape as the states.
        :param mask: boolean, optional, required with x. Which state indexes are updatable by x
        :return:
        """
        h = self.states - 1  # + torch.normal(0, self.noise, self.states.shape)  # inject noise (and subtract 1?)
        self.edge.update(h)  # do local weight update
        out_activ = self.edge(h).clone()  # get output from all edges.

        if x is not None:
            if x.shape == self.states.shape:
                if self.input_mode == "overwrite":
                    # used state values should be != 0.
                    out_activ = (out_activ * torch.logical_not(mask)) + x
                elif self.input_mode == "additive":
                    out_activ = out_activ + x
            else:
                raise IndexError
        # mix state update and current state values parameterized by resistance.
        self.states = self.states * self.resistance + out_activ
        if self.past_states is not None:
            self.past_states.append(self.states.clone())
        return self.states

    def detach(self, reset_intrinsic=False):
        # detach computational graph
        self.edge.detach(reset_weight=reset_intrinsic)
        states = torch.zeros_like(self.states)
        self.states = torch.nn.init.xavier_normal_(states)
        self.past_states = []
        return self

    def parameters(self, recurse: bool = True):
        """
        :return: list of parameters that can be optimized by gradient descent.
        """
        params = self.edge.parameters() + [self.resistance]
        return params

    def set_grad(self, grads):
        self.resistance.grad = grads[-1]
        self.edge.set_grad(grads[:-1])

    def l1(self):
        ps = self.parameters()
        penalty = torch.sum(torch.stack([torch.sum(torch.abs(p)) for p in ps]))
        return penalty

    def clone(self, fuzzy=False, device=None):
        if device is None:
            device = self.device
        new_model = FCIntrinsic(self.num_nodes, (1, self.edge.channels, self.edge.spatial),
                              inject_noise=self.inject_noise, edge_module=FCPlasticEdges, device=device,
                              track_activation_history=self.past_states is not None, mask=self.edge.mask,
                              is_resistive=self.resistive, input_mode=self.input_mode, optimize_weights=self.edge.optimize_weights)
        new_model.states = self.states.detach().to(device)
        new_model.edge = self.edge.clone(fuzzy=fuzzy).to(device)
        new_model.resistance = torch.nn.Parameter(self.resistance.detach().clone().to(device))
        return new_model

    def to(self, device):
        self.device = device
        return self.clone(fuzzy=False, device=device)
