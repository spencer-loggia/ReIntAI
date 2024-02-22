import torch
import numpy as np
from module import PlasticEdges


class ElegantReverbNetwork(torch.nn.Module):
    """
    Parallelized model.
    """
    def __init__(self, num_nodes, node_shape: tuple = (1, 3, 64, 64), inject_noise=True,
                 edge_module=PlasticEdges, device='cpu', track_activation_history=False, input_nodes=(1,),
                 mask=None, kernel_size=3, is_resistive=True):
        """

        :param num_nodes:
        :param node_shape:
        :param inject_noise:
        :param edge_module:
        :param device:
        :param track_activation_history:
        :param input_nodes: if input nodes is None, stim inputs to all nodes. Otherwise, mask is set to only project ot input nodes.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.resistive = is_resistive
        if self.resistive:
            self.resistance = 0.33
        else:
            self.resistance = 0.0
        states = torch.empty(size=(self.num_nodes, node_shape[1], node_shape[2], node_shape[3]), device=device)
        self.states = torch.nn.init.xavier_normal_(states)
        if mask is None:
            mask = torch.ones((num_nodes, num_nodes), device=device)
        if input_nodes is not None:
            input_nodes = np.array(input_nodes)  # compensate for added stim node.
        # synaptic module takes n x c x s1 x s2 input and returns output of the same shape.
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
        if x is not None:
            if x.shape == self.states.shape:
                self.states = (self.states * torch.logical_not(x)) + x
            else:
                raise IndexError
        activ = self.sigmoid(self.states - 1.0 + torch.normal(0, self.noise, self.states.shape))
        #activ = (self.states > 1.).float()
        self.edge.update(activ)
        out_activ = self.edge(activ).clone()
        self.states = self.resistance * self.states.clone() + out_activ
        if self.past_states is not None:
            self.past_states.append(self.states.clone())
        return self.states

    def detach(self, reset_intrinsic=False):
        self.edge.detach(reset_weight=reset_intrinsic)
        states = torch.zeros_like(self.states)
        self.states = torch.nn.init.xavier_normal_(states)
        self.past_states = []
        return self

    def parameters(self, recurse: bool = True):
        params = self.edge.parameters()
        return params

    def l1(self):
        return torch.sum(torch.abs(self.edge.out_edge))