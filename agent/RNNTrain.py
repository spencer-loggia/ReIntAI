import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import adam
import numpy as np


from agent.agents import WaterworldAgent

2*32 size
activation funtion: tanh


class RNNWaterworldAgent(WaterworldAgent):

    def __init__(self, num_layers, num_nodes, device='cpu', input_mode="overwrite", *arg, **kwargs):

        super().__init__(*arg, **kwargs)

        if "decode_node" in kwargs:
            self.decode_node = kwargs["decode_node"]
        else:
            self.decode_node = 2
        if self.decode_node is None:
            self.value_decoder = torch.empty((self.input_size, 1), device=self.device)
            self.value_decoder = torch.nn.Parameter(torch.nn.init.xavier_normal_(self.value_decoder))
        else:
            self.value_decoder = torch.empty((self.spatial, 1), device=self.device)
            self.value_decoder = torch.nn.Parameter(torch.nn.init.xavier_normal_(self.value_decoder))
        self.value_decoder_bias = torch.nn.Parameter(torch.zeros((1,), device=self.device) + .001)

        self.policy_decoder = torch.empty((self.spatial, 4), device=self.device)
        self.policy_decoder = torch.nn.Parameter(torch.nn.init.xavier_normal_(self.policy_decoder))
        self.policy_decoder_bias = torch.nn.Parameter(torch.zeros((4,), device=self.device) + .001)

        input_encoder = torch.empty((self.input_size, self.spatial * self.input_channels), device=self.device)
        input_encoder = torch.nn.init.xavier_normal_(input_encoder)

        self.input_encoder = torch.nn.Parameter(input_encoder)
        self.input_encoder_bias = torch.nn.Parameter(torch.zeros((1,), device=self.device) + .001)


        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.device = device
        self.input_mode = input_mode


        self.rnn1 = nn.RNN(input_size, num_nodes, num_layers=1, batch_first=True)
            # I need to figure out the input dimention of X (observation space except for velocities)
        self.rnn2 = nn.RNN(num_nodes, num_nodes, num_layers=1, batch_first=True)

        for param in self.rnn1.parameters():
            if len(param.shape) >= 2:  # Apply Xavier to weight matrices only
                nn.init.xavier_uniform_(param.data)
        for param in self.rnn2.parameters():
            if len(param.shape) >= 2:  # Apply Xavier to weight matrices only
                nn.init.xavier_uniform_(param.data)


