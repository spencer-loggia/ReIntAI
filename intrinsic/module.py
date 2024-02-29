import torch
from intrinsic import util


class PlasticEdges(torch.nn.Module):
    def __init__(self, num_nodes, spatial1, spatial2, kernel_size, channels, device='cpu',
                 normalize_conv=True, mask=None, optimize_weights=True, debug=False, **kwargs):
        """
        designed to operate on a graph all at once.
        """
        super().__init__()
        # The activation memory tracks the last state of the model. It is necessary for for computing the instrisic edge
        # update function
        self.activation_memory = None
        self.num_nodes = num_nodes
        self.kernel_size, self.pad = util.conv_identity_params(in_spatial=spatial1, desired_kernel=kernel_size)
        self.channels = channels
        self.spatial1 = spatial1
        self.spatial2 = spatial2

        # mask has shape (nodes, nodes) and allows us to predefine a graph structure besides fully connected.
        if mask is None:
            mask = torch.ones((num_nodes, num_nodes), device=device)
        self.mask = mask

        # Non-Parametric Weights used for intrinsic update
        init_weight = torch.empty((num_nodes, num_nodes,
                                   1, 1,
                                   channels, channels, self.kernel_size, self.kernel_size),
                                   device=device)  # 8D Tensor.
        self.init_weight = torch.nn.Parameter(torch.nn.init.xavier_normal_(init_weight))

        self.weight = self._expand_base_weights(self.init_weight)

        # Channel Mapping
        chan_map = torch.empty((num_nodes, num_nodes, channels, channels), device=device)
        self.chan_map = torch.nn.Parameter(torch.nn.init.xavier_normal_(chan_map))

        if "init_plasticity" in kwargs:
            init_plasticity = kwargs["init_plasticity"]
        else:
            init_plasticity = .1
        self.plasticity = torch.nn.Parameter(torch.ones((num_nodes, num_nodes), device=device) * init_plasticity)
        self.device = device
        self.unfolder = torch.nn.Unfold(kernel_size=self.kernel_size, padding=self.pad)
        self.folder = torch.nn.Fold(kernel_size=self.kernel_size,
                                    output_size=(spatial1, spatial2),
                                    padding=self.pad)
        self.normalize = normalize_conv
        self.debug = debug

    def _expand_base_weights(self, in_weight):
        # adds explicit spatial dims to weights
        expanded_weights = torch.tile(in_weight.clone(), (1, 1, self.spatial1, self.spatial2, 1, 1, 1, 1))
        return expanded_weights

    def parameters(self):
        params = [self.chan_map, self.plasticity, self.init_weight]
        return params

    def forward(self, x):
        x = x.to(self.device)  # nodes, channels, spatial1, spatial2
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
        combined_weight = combined_weight * self.chan_map.view((self.num_nodes, self.num_nodes, 1, self.channels, self.channels, 1))

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

    def get_weight(self):
        return self.out_edge

    def update(self, target_activation):
        """
        intrinsic update
        :param target_activation: the activation of each state after forward pass. (nodes, channel, spatial, spatial)
        :param args:
        :return:
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
        target_activations = target_activation.view(1, self.num_nodes, self.channels,
                                                    self.spatial1 * self.spatial2).transpose(2, 3) # (_, n, s, c)

        # TODO:: This should be refactored to use torch.linalg.lstsq for better performance.
        reverse_conv = torch.linalg.pinv(self.chan_map.view((self.num_nodes, self.num_nodes, self.channels, self.channels)))

        # reverse_conv = self.chan_map.clone().transpose(2, 3)  # (node, node, in_chan, out_chan)
        iter_rule = "uvsc, uvoc -> uvsc"
        target_meta_activations = torch.einsum(iter_rule, target_activations, reverse_conv).transpose(2,
                                                                                                      3)  # source, target, channels, spatial
        target_meta_activations = target_meta_activations.view(self.num_nodes * self.num_nodes, self.channels,
                                                               self.spatial1, self.spatial2)
        ufld_target = self.unfolder(target_meta_activations).transpose(1,
                                                                       2)  # nodes * nodes, spatial1 * spatial2, channels * kernel * kernel
        ufld_target = ufld_target.view((self.num_nodes, self.num_nodes, self.spatial1 * self.spatial2, self.channels,
                                        self.kernel_size * self.kernel_size))

        coactivation = self.activation_memory * ufld_target  # [0, 1] -> [0, 1]

        plasticity = self.plasticity.view(self.num_nodes, self.num_nodes, 1, 1, 1, 1, 1).clone()
        if self.debug:
            plasticity.register_hook(lambda grad: print("plast", grad.reshape(grad.shape[0], -1).sum(dim=-1)))
        self.weight = (1 - plasticity) * self.weight + plasticity * coactivation.view((self.num_nodes, self.num_nodes,
                                                                                       self.spatial1, self.spatial2,
                                                                                       self.channels, self.kernel_size,
                                                                                       self.kernel_size))
        return

    def detach(self, reset_weight=False):
        if reset_weight:
            self.weight = self._expand_base_weights(self.init_weight)
        else:
            self.weight = self.weight.clone()
        # self.chan_map = self.chan_map.detach()
        # self.plasticity = self.plasticity.detach()
        self.activation_memory = None

    def to(self, device):
        self.init_weight = torch.nn.Parameter(self.init_weight.to(device))
        self.weight = self.weight.to(device)
        self.chan_map = torch.nn.Parameter(self.chan_map.to(device))
        self.plasticity = torch.nn.Parameter(self.plasticity.to(device))
        self.device = device
        return self
