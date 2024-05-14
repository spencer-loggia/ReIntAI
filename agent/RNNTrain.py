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
from intrinsic.model import Intrinsic
from reward_functions import ActorCritic
from exist import episode
from evolve import EvoController


class RNN_Agent(WaterworldAgent):

    def __init__(self, num_nodes, channels, spatial, kernel, device, *arg, **kwargs):

        super().__init__(num_nodes = num_nodes, channels = channels, spatial = spatial, kernal = kernel, *arg, **kwargs)

        if "decode_node" in kwargs:
            self.decode_node = kwargs["decode_node"]
        else:
            self.decode_node = 2
        if self.decode_node is None:
            self.value_decoder = torch.empty((num_nodes, 1), device=self.device) # We need to double check this line about the input_size
            self.value_decoder = torch.nn.Parameter(torch.nn.init.xavier_normal_(self.value_decoder))
        else:
            self.value_decoder = torch.empty((num_nodes, 1), device=self.device)
            self.value_decoder = torch.nn.Parameter(torch.nn.init.xavier_normal_(self.value_decoder))
        self.value_decoder_bias = torch.nn.Parameter(torch.zeros((1,), device=self.device) + .001)

        self.policy_decoder = torch.empty((num_nodes, 4), device=self.device)
        self.policy_decoder = torch.nn.Parameter(torch.nn.init.xavier_normal_(self.policy_decoder))
        self.policy_decoder_bias = torch.nn.Parameter(torch.zeros((1,), device=self.device) + .001)

        input_encoder = torch.empty((self.input_size, num_nodes), device=self.device)
        input_encoder = torch.nn.init.xavier_normal_(input_encoder)

        self.input_encoder = torch.nn.Parameter(input_encoder)
        self.input_encoder_bias = torch.nn.Parameter(torch.zeros((1,), device=self.device) + .001)

        self.num_nodes_rnn = num_nodes
        self.device = device

        self.rnn = nn.RNN(num_nodes, num_nodes, num_layers=2, batch_first=True)
        # self.rnn2 = nn.RNN(num_nodes, num_nodes, num_layers=1, batch_first=True)

        for param in self.rnn.parameters():
            if len(param.shape) >= 2:  # Apply Xavier to weight matrices only
                nn.init.xavier_uniform_(param.data)

        core_model = Intrinsic(num_nodes)

    def set_grad(self, grad):

        self.input_encoder.grad = grad[0]
        self.input_encoder_bias.grad = grad[1]
        self.value_decoder.grad = grad[2]
        self.value_decoder_bias.grad = grad[3]
        self.policy_decoder.grad = grad[4]
        self.policy_decoder_bias.grad = grad[5]
        self.core_model.set_grad(grad[6:])
        # self.rnn.bias_hh_l0.data.grad = grad[6]

    def forward(self, X, r=None):

        # Encode input using input_encoder
        encoded_input = (X + self.input_encoder_bias) @ self.input_encoder
        encoded_input = encoded_input.view((-1, 1, self.num_nodes_rnn))  # Reshape for RNN input

        # Pass through first RNN layer
        rnn_output, _ = self.rnn(encoded_input)

        value_est = (rnn_output[0, 0, :] + self.value_decoder_bias) @ self.value_decoder
        action_est = (rnn_output[0, 0, :] + self.policy_decoder_bias) @ self.policy_decoder

        act_fxn = torch.exp(action_est)

        # Split action_params into meaningful parts (mu and sigma)
        c1 = act_fxn[0:2]  # Applying tanh to scale outputs as needed
        c2 = act_fxn[2:]  # Ensure sigma is positive

        return c1, c2, value_est

# model = RNNtrain(num_nodes=36, device='cpu', input_mode='overwrite')

# def train(agent, epochs, device, learning_rate):
#
#     # EPOCHS = 20000
#     # EPSILON_START = 1.0
#     # EPSILON_DECAY = 4000
#     # ALGORITM = "a3c"
#     # LOG_MAX_LR = -3
#     # LOG_MIN_LR = -8
#     # evo = EvoController(seed_agent=agent, epochs=EPOCHS, num_base=5, num_workers=10,
#     #                     min_agents=1, max_agents=3, min_gen=1, max_gen=1, log_min_lr=LOG_MIN_LR,
#     #                     log_max_lr=LOG_MAX_LR,
#     #                     algo=ALGORITM, start_epsilon=EPSILON_START, inverse_eps_decay=EPSILON_DECAY,
#     #                     worker_device="cpu")
#     copies = [1]
#     base_agents=[agent]
#
#     for epoch in range(epochs):
#
#         gen_info, base_scores = episode(base_agents, copies)
#         agent_info = gen_info['pursuer_0']
#         actor_critic = ActorCritic(gamma=0.96, alpha=0.4)
#         val_loss, policy_loss = actor_critic.loss(torch.tensor(agent_info["inst_r"], device=device),
#                                                      torch.concat(agent_info["value"], dim=0),
#                                                      torch.stack(agent_info["action_likelihood"],dim=0),
#                                                      torch.stack(agent_info["entropy"], dim=0))
#         total_loss = val_loss + policy_loss
#         total_loss.backward()
#
#         optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
#         optimizer.step()
#
#         print(epoch + 1, "/", epochs)
#         print("Loss: ", total_loss.item())

# model = RNN_Agent(num_nodes=36, num_layers=2, device="cpu", input_mode="override")
# train(model, 20, "cpu", 0.001)