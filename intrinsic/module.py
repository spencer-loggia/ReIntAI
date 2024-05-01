import random

import torch
from intrinsic import util


class PlasticEdges():
    def __init__(self, num_nodes, spatial1, spatial2, kernel_size, channels, device='cpu',
                 mask=None, optimize_weights=True, debug=False, **kwargs):
        """
        Designed to operate on a (n, c, s, s) intrinsic graph. Defines a convolutional edge with a Hebbian-like
        local update function between each node and each channel on the graph.
        :param num_nodes: The number of nodes in the intrinsic graph.
        :param spatial1: The first spatial dimension
        :param spatial2: The second spatial dimension (should be equal to spatial1 in current version)
        :param kernel_size: The convolution kernel size for the edges. Must be less than min(spatial1, spatial2)
        :param channels: The number of channels in the nodes.
        :param device: The device to preform computations on. (cpu, cuda0...n, tpu0...n)
        :param mask: A user defined mask as a (n, n) adj matrix to modify node to node weights.
        :param optimize_weights: Whether to fit the initial convolutional weights using gradient decent.
        :param debug: Whether to print info about gradients and current states at runtime
        :param kwargs: addition keyword arguments.
        """
        # The activation memory tracks the last state of the model. It is necessary for computing the intrinsic edge
        # update function
        self.activation_memory = None
        self.optimize_weights = optimize_weights
        self.num_nodes = num_nodes
        self.kernel_size, self.pad = util.conv_identity_params(in_spatial=spatial1, desired_kernel=kernel_size)
        self.channels = channels
        self.spatial1 = spatial1
        self.spatial2 = spatial2

        # mask has shape (nodes, nodes) and allows us to predefine a graph structure besides fully connected.
        if mask is None:
            mask = torch.ones((num_nodes, num_nodes), device=device)
        self.mask = mask

        # initial weight parameter
        self.init_weight = torch.zeros((num_nodes, num_nodes,
                                   1, 1,
                                   self.channels, self.channels, self.kernel_size, self.kernel_size),
                                   device=device)  # 8D Tensor.
        self.init_weight = torch.nn.init.xavier_normal_(self.init_weight * .1)
        if optimize_weights:
            self.init_weight = torch.nn.Parameter(self.init_weight)

        # Non-Parametric Weights used for intrinsic update
        self.weight = None  # (n, n, s, s, c, c, k, k)

        # Channel Mapping
        chan_map = torch.empty((num_nodes, num_nodes, channels, channels), device=device)
        self.chan_map = torch.nn.Parameter(torch.nn.init.xavier_normal_(chan_map * .0001))

        if "init_plasticity" in kwargs:
            init_plasticity = kwargs["init_plasticity"]
        else:
            init_plasticity = .1
        self.plasticity = torch.nn.Parameter(torch.ones((num_nodes, num_nodes, channels, channels), device=device) * init_plasticity)
        self.device = device
        self.unfolder = torch.nn.Unfold(kernel_size=self.kernel_size, padding=self.pad)
        self.folder = torch.nn.Fold(kernel_size=self.kernel_size,
                                    output_size=(spatial1, spatial2),
                                    padding=self.pad)
        self.debug = debug

    def _expand_base_weights(self, in_weight):
        # adds explicit spatial dims to weights
        if self.init_weight.shape[4] == 1:
            expanded_weights = torch.sigmoid(torch.tile(in_weight.clone(), (1, 1, self.spatial1, self.spatial2, self.channels, self.channels, 1, 1)))
        else:
            expanded_weights = torch.sigmoid(torch.tile(in_weight.clone(), (1, 1, self.spatial1, self.spatial2, 1, 1, 1, 1)))
        return expanded_weights

    def parameters(self):
        params = [self.chan_map, self.plasticity, self.init_weight]
        return params

    def set_grad(self, grads):
        self.chan_map.grad = grads[0]
        self.plasticity.grad = grads[1]
        self.init_weight.grad = grads[2]

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        The forward pass on all edges. Takes (n, c, s, s) state as input, computes activation function on it, and sends
        through the current weight matrix and channel map matrix. Returns a state update matrix with the same shape ad
        the input, i.e. (n, c, s, s).
        :param x: Tensor, Input intrinsic graph states (n, c, s, s)
        :return: Tensor, update in the same space as x - (n, c, s, s)
        """
        if self.weight is None:
            self.weight = self._expand_base_weights(self.init_weight)
        x = x.to(self.device)  # nodes, channels, spatial1, spatial2
        x = torch.sigmoid(x) # compute sigmoid activation on range [0, 1]
        if len(x.shape) != 4:
            raise ValueError("Input Tensor Must Be 4D, not shape", x.shape)
        if x.shape[0] != self.num_nodes:
            raise ValueError("Input Tensor must have number of nodes on batch dimension.")
        if (torch.max(x) > 1 or torch.min(x) < 0) and self.debug:
            print("WARN: Reverb  input activations are expected to have range 0 to 1")
        xufld = self.unfolder(x).transpose(1, 2)  # nodes, spatial1 * spatial2, channels * kernel * kernel
        xufld = xufld.view((self.num_nodes, 1, self.spatial1 * self.spatial2, self.channels, self.kernel_size ** 2))
        # unfolded states will broadcast over input node dim.

        self.activation_memory = xufld.clone()

        # weights are zeroed for node -> node maps that are masked.
        combined_weight = self.weight * self.mask.view(self.num_nodes, self.num_nodes, 1, 1, 1, 1, 1, 1)
        combined_weight = combined_weight.view(
            (self.num_nodes, self.num_nodes, self.spatial1 * self.spatial2, self.channels, self.channels,
             self.kernel_size ** 2))

        # Compose plastic weights and channel map
        # uvscok,
        # combined_weight = combined_weight * self.chan_map.view((self.num_nodes, self.num_nodes, 1, self.channels, self.channels, 1))
        iterrule = "uvscok, vbop -> ubscpk"
        combined_weight = torch.einsum(iterrule, combined_weight, self.chan_map)

        # src_nodes (u), target_node (v), flat_spatial (s), channels (c), flat_kernel (k)
        # src_nodes (u), target_node (v), flat_spatial (s), in_channels (c), out_channels (o), flat_kernel (k)
        # this einsum will sum across source node dimension and map channels in parallel.
        iter_rule = "uvsck, uvscok -> vsok"
        mapped_meta = torch.einsum(iter_rule, xufld, combined_weight)
        if self.debug:
            mapped_meta.register_hook(lambda grad: print("post_einsum", grad.reshape(grad.shape[0], -1).sum(dim=-1)))

        ufld_meta = mapped_meta.transpose(2,3)  # switch the ordering of kernels and channels to original so we can take the correct view on them
        ufld_meta = ufld_meta.reshape(
            (self.num_nodes, self.spatial1 * self.spatial2, self.kernel_size ** 2 * self.channels)
        ).transpose(1, 2)  # finish returning to original unfolded

        # fold up to state space (sum unit receptive fields)
        out = self.folder(ufld_meta)  # nodes, channels, spatial, spatial
        if self.debug:
            out.register_hook(lambda grad: print("out", grad.reshape(grad.shape[0], -1).sum(dim=-1)))
        return out

    def update(self, target_activation):
        """
        Compute and apply the local hebbian like update for the weight matrix. At a high level, weights that connect
        units that have high (near 1) activations at adjacent time steps should increase, and weights that connect units
        with low or uneven connections at adjacent time sets should decrease.
        :param target_activation: the value of each state after forward pass before activation (nodes, channel, spatial, spatial)
        :return: None
        """

        if len(target_activation.shape) != 4:
            raise ValueError("Input Tensor Must Be 4D, not shape", target_activation.shape)
        if target_activation.shape[0] != self.num_nodes:
            raise ValueError("Input Tensor must have number of nodes on batch dimension.")
        if self.debug and (torch.max(target_activation) > 1 or torch.min(target_activation) < 0):
            print("WARN: Reverb  input activations are expected to have range 0 to 1")
        if self.activation_memory is None:
            return

        # shape of chanel view of synaptic unfolded space
        # channel_view = (self.kernel_size ** 2, self.in_channels, self.spatial1, self.spatial2)

        # reverse the channel mapping so source channels receive information about their targets
        target_activations = target_activation.view(self.num_nodes * self.channels,
                                                    self.spatial1 * self.spatial2, 1).transpose(0, 1)  # (_, s, nc)

        chan_map = self.chan_map.permute((0, 2, 1, 3)).reshape((1, self.num_nodes * self.channels, self.num_nodes * self.channels))

        # This is a fast and  numerically stable equivalent to (chan_map ^ -1) (target_activations)
        target_meta_activations = torch.sigmoid(torch.linalg.solve(chan_map, target_activations))
        target_meta_activations = target_meta_activations.transpose(0, 1).reshape(self.num_nodes, self.channels, self.spatial1, self.spatial2) # u, c, s, s

        # unfold the current remapped activations
        ufld_target = self.unfolder(target_meta_activations).transpose(1, 2) # nodes, spatial1 * spatial2, channels * kernel * kernel
        ufld_target = ufld_target.view((self.num_nodes, self.spatial1**2, self.channels,
                                        self.kernel_size**2))

        activ_mem = self.activation_memory.view((self.num_nodes, self.spatial1**2, self.channels,
                                                 self.kernel_size**2))

        # This is an outer product on the channel dimension and elementwise on all others.
        iterrule = "usck, vsok -> uvscok"
        #coactivation = torch.exp(torch.einsum(iterrule, activ_mem, ufld_target))
        coactivation = torch.exp(torch.einsum(iterrule, activ_mem, ufld_target))
        plasticity = self.plasticity.view(self.num_nodes, self.num_nodes, 1, 1, self.channels, self.channels, 1, 1).clone()

        if self.debug:
            plasticity.register_hook(lambda grad: print("plast", grad.reshape(grad.shape[0], -1).sum(dim=-1)))
        self.weight = torch.log((1 - plasticity) * torch.exp(self.weight) + plasticity * coactivation.view((self.num_nodes, self.num_nodes,
                                                                self.spatial1, self.spatial2,
                                                                self.channels, self.channels,
                                                                self.kernel_size, self.kernel_size)))
        return

    def instantiate(self):
        instance = PlasticEdges(self.num_nodes, self.spatial1, self.spatial2, self.kernel_size, self.channels,
                                device=self.device, mask=self.mask, optimize_weights=self.optimize_weights,
                                debug=self.debug)
        instance.init_weight = self.init_weight.clone()
        instance.weight = instance._expand_base_weights(instance.init_weight)
        instance.chan_map = self.chan_map.clone()
        instance.plasticity = self.plasticity.clone()
        return instance

    def detach(self, reset_weight=False):
        if reset_weight:
            self.weight = None
        else:
            if self.weight is not None:
                self.weight = self.weight.detach().clone()
        self.activation_memory = None
        return self

    def to(self, device):
        if self.optimize_weights:
            self.init_weight = torch.nn.Parameter(self.init_weight.to(device))
        else:
            self.init_weight = self.init_weight.to(device)
        self.weight = self.weight.to(device)
        self.chan_map = torch.nn.Parameter(self.chan_map.to(device))
        self.plasticity = torch.nn.Parameter(self.plasticity.to(device))
        self.device = device
        return self

    def clone(self, fuzzy=False):
        instance = PlasticEdges(self.num_nodes, self.spatial1, self.spatial2, self.kernel_size, self.channels,
                                device=self.device, mask=self.mask, optimize_weights=self.optimize_weights,
                                debug=self.debug)
        if fuzzy:
            s1 = float(self.init_weight.std()) * (.5 * random.random() + .1)
            s2 = float(self.chan_map.std()) * (.5 * random.random() + .1)
            s3 = float(self.plasticity.std()) * (.5 * random.random() + .1)
            m1 = m2 = m3 = 0.
        else:
            s1 = s2 = s3 = m1 = m2 = m3 = 0.

        instance.init_weight = torch.nn.Parameter(self.init_weight.detach().clone() + torch.normal(size=self.init_weight.shape,
                                                                       mean=m1,
                                                                       std=s1))

        instance.weight = instance._expand_base_weights(instance.init_weight)
        instance.chan_map = torch.nn.Parameter(self.chan_map.detach().clone() + torch.normal(size=self.chan_map.shape,
                                                                 mean=m2,
                                                                 std=s2))
        instance.plasticity = torch.nn.Parameter(self.plasticity.detach().clone() + torch.normal(size=self.chan_map.shape,
                                                                     mean=m3,
                                                                     std=s3))
        return instance
