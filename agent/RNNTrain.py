import copy
import math
import pickle
import random
import sys
import timeit
from typing import List
import randomname
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import multiprocessing as mp
import matplotlib.pyplot as plt

from agents import WaterworldAgent
from reward_functions import ActorCritic
from exist import episode

class RNNtrain(WaterworldAgent):

    def __init__(self, num_nodes = 36, device='cpu', input_mode="overwrite", *arg, **kwargs):

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

        self.num_nodes = num_nodes
        self.device = device
        self.input_mode = input_mode

        self.rnn1 = nn.RNN(self.input_size, num_nodes, num_layers=1, batch_first=True)
        self.rnn2 = nn.RNN(num_nodes, num_nodes, num_layers=1, batch_first=True)

        for param in self.rnn1.parameters():
            if len(param.shape) >= 2:  # Apply Xavier to weight matrices only
                nn.init.xavier_uniform_(param.data)
        for param in self.rnn2.parameters():
            if len(param.shape) >= 2:  # Apply Xavier to weight matrices only
                nn.init.xavier_uniform_(param.data)

    def forward(self, X, r=None):

        # Encode input using input_encoder
        encoded_input = (X + self.input_encoder_bias) @ self.input_encoder
        encoded_input = encoded_input.view(-1, 1, self.input_size)  # Reshape for RNN input

        # Pass through first RNN layer
        rnn_output1, _ = self.rnn1(encoded_input)

        # Pass through second RNN layer
        rnn_output2, _ = self.rnn2(rnn_output1)

        # Decode the output to compute action parameters and value estimates
        # Assuming the last output of the RNN contains the features for decoding
        last_rnn_output = rnn_output2[:, -1, :]  # Taking the output of the last time step

        action_params = (last_rnn_output @ self.policy_decoder) + self.policy_decoder_bias
        value_est = (last_rnn_output @ self.value_decoder) + self.value_decoder_bias

        # Split action_params into meaningful parts (mu and sigma)
        mu = torch.tanh(action_params[..., :2])  # Applying tanh to scale outputs as needed
        sigma = torch.exp(action_params[..., 2:])  # Ensure sigma is positive

        return mu, sigma, value_est
    def instantiate(self):
        action_likelihood = []
        entropy = []
        values =[]
        rewards = []
        new_agent = WaterworldAgent(num_nodes=self.core_model.num_nodes,
                                    channels=self.channels, spatial=self.spatial, sensors=self.num_sensors,
                                    action_dim=self.action_dim,
                                    device=self.device, input_channels=self.input_channels, decode_node=self.decode_node)
        # new_core = self.core_model.instantiate()
        # new_agent.core_model = new_core
        c1, c2, value_estimates = self.forward(X)
        x_dist = torch.distributions.Beta(concentration0=c1[0], concentration1=c1[1])
        y_dist = torch.distributions.Beta(concentration0=c2[0], concentration1=c2[1])
        action_x = x_dist.sample()
        action_y = y_dist.sample()
        likelihood_x = x_dist.log_prob(action_x)
        likelihood_y = y_dist.log_prob(action_y)
        action_likelihood.append(likelihood_x+likelihood_y)
        entropy.append (x_dist.entropy() + y_dist.entropy())
        values.append (value_estimates)
        actor_critic = ActorCritic(gamma=0.96, alpha=0.4)
        valuesTensor = torch.FloatTensor(values)
        val_loss, policy_loss = actor_critic.loss(torch.tensor(rewards, device=device),
                                                     torch.concat(valuesTensor, dim=0),
                                                     torch.stack(torch.tensor(action_likelihood).float ,dim=0),
                                                     torch.stack(torch.tensor(entropy).float ,dim=0))
        total_loss = val_loss + policy_loss
        total_loss.backward()

        new_agent.policy_decoder = self.policy_decoder.clone()
        new_agent.value_decoder = self.value_decoder.clone()
        new_agent.input_encoder = self.input_encoder.clone()
        new_agent.policy_decoder_bias = self.policy_decoder_bias.clone()
        new_agent.value_decoder_bias = self.value_decoder_bias.clone()
        new_agent.input_encoder_bias = self.input_encoder_bias.clone()
        new_agent.epsilon = self.epsilon
        new_agent.id = self.id
        return new_agent
