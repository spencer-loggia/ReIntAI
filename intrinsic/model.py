import torch
import numpy as np
from intrinsic.module import PlasticEdges


class Intrinsic(torch.nn.Module):
    """
    Parallelized model.
    """
    def __init__(self, num_nodes, node_shape: tuple = (1, 3, 64, 64), inject_noise=True,
                 edge_module=PlasticEdges, device='cpu', track_activation_history=False, input_nodes=(1,),
                 mask=None, kernel_size=3, is_resistive=True, input_mode="additive"):
        """
        :param num_nodes: Number of nodes in the graph.
        :param node_shape: Shape (channels and spatial of each node in the graph.
        :param inject_noise: Whether to inject random noise at each step.
        :param edge_module: Class to use for edges.
        :param device: Hardware device to use for computation.
        :param input_mode: How inputs should be injected into the graph.
        :param track_activation_history: Whether to store the state history
        :param input_nodes: if input nodes is None, stim inputs to all nodes. Otherwise, mask is set to only project
        ot input nodes.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.resistive = is_resistive
        if input_mode not in ["additive", "overwrite"]:
            raise ValueError
        self.input_mode = input_mode
        if self.resistive:
            self.resistance = 0.33
        else:
            self.resistance = 0.0
        # initialization of state matrix
        states = torch.empty(size=(self.num_nodes, node_shape[1], node_shape[2], node_shape[3]), device=device)
        self.states = torch.nn.init.xavier_normal_(states)
        # optional mask to ablate some connections,
        if mask is None:
            mask = torch.ones((num_nodes, num_nodes), device=device)

        # edge module takes n x c x s1 x s2 input and returns output of the same shape.
        self.edge = edge_module(self.num_nodes, node_shape[2], node_shape[3], kernel_size=kernel_size, channels=node_shape[1],
                                device=device, mask=mask, inject_noise=inject_noise, normalize_conv=False, init_plasticity=.2)
        self.inject_noise = inject_noise
        self.noise = .1
        self.sigmoid = torch.nn.Sigmoid()
        if track_activation_history:
            self.past_states = []
        else:
            self.past_states = None
        self.device = device

    def forward(self, x=None):
        """
        :param x: optional. Tensor with the same shape as the states.
        :return:
        """
        h = self.states - 1.0 + torch.normal(0, self.noise, self.states.shape)  # inject noise (and subtract 1?)
        self.edge.update(h)  # do local update
        out_activ = self.edge(h).clone()  # get output from all edges.

        if x is not None:
            if x.shape == self.states.shape:
                if self.input_mode == "overwrite":
                    # used state values should be != 0.
                    out_activ = (out_activ * torch.logical_not(x)) + x
                elif self.input_mode == "additive":
                    out_activ = out_activ + x
            else:
                raise IndexError

        self.states = self.resistance * (self.states + out_activ)
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
        params = self.edge.parameters()
        return params

    def l1(self):
        return torch.sum(torch.abs(self.edge.out_edge))