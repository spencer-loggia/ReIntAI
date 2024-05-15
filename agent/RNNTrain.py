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
from pettingzoo.sisl import waterworld_v4
from intrinsic.model import Intrinsic
from reward_functions import ActorCritic
from exist import episode
from evolve import EvoController
from intrinsic.module import PlasticEdges


class RNN_Agent(WaterworldAgent):

    def __init__(self, num_nodes, device, *arg, **kwargs):

        super().__init__(num_nodes = num_nodes,  *arg, **kwargs)

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

        # core_model = Intrinsic(num_nodes)

    def parameters(self):
        agent_heads = [self.input_encoder, self.input_encoder_bias, self.value_decoder,
                       self.value_decoder_bias, self.policy_decoder, self.policy_decoder_bias, self.rnn.bias_hh_l0,
                       self.rnn.bias_hh_l1, self.rnn.bias_ih_l0, self.rnn.bias_ih_l1, self.rnn.weight_hh_l0, self.rnn.weight_hh_l1,
                       self.rnn.weight_ih_l0, self.rnn.weight_ih_l1]
        return agent_heads

    def set_grad(self, grad):

        self.input_encoder.grad = grad[0]
        self.input_encoder_bias.grad = grad[1]
        self.value_decoder.grad = grad[2]
        self.value_decoder_bias.grad = grad[3]
        self.policy_decoder.grad = grad[4]
        self.policy_decoder_bias.grad = grad[5]
        self.rnn.bias_hh_l0.grad = grad[6]
        self.rnn.bias_hh_l1.grad = grad[7]
        self.rnn.bias_ih_l0.grad = grad[8]
        self.rnn.bias_ih_l1.grad = grad[9]
        self.rnn.weight_hh_l0.grad = grad[10]
        self.rnn.weight_hh_l1.grad = grad[11]
        self.rnn.weight_ih_l0.grad = grad[12]
        self.rnn.weight_ih_l1.grad = grad[13]

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

# model = RNN_Agent(num_nodes=36, num_layers=2, device="cpu")

    def clone(self, fuzzy=True):

        new_agent = RNN_Agent(num_nodes=36, num_layers=2, channels=1, spatial=5, kernel=None, device="cpu")

        if not fuzzy:
            new_agent.id = self.id
            new_agent.version = self.version
            new_agent.fitness = self.fitness
            new_agent.v_loss = self.v_loss
            new_agent.p_loss = self.p_loss
        new_agent.epsilon = self.epsilon

        with torch.no_grad():

            new_agent.rnn.bias_hh_l0 = torch.nn.Parameter(self.rnn.bias_hh_l0.detach().clone())
            new_agent.rnn.bias_hh_l1 = torch.nn.Parameter(self.rnn.bias_hh_l1.detach().clone())
            new_agent.rnn.bias_ih_l0 = torch.nn.Parameter(self.rnn.bias_hh_l0.detach().clone())
            new_agent.rnn.bias_ih_l1 = torch.nn.Parameter(self.rnn.bias_hh_l1.detach().clone())
            new_agent.rnn.weight_hh_l0 = torch.nn.Parameter(self.rnn.weight_hh_l0.detach().clone())
            new_agent.rnn.weight_hh_l1 = torch.nn.Parameter(self.rnn.weight_hh_l1.detach().clone())
            new_agent.rnn.weight_ih_l0 = torch.nn.Parameter(self.rnn.weight_ih_l0.detach().clone())
            new_agent.rnn.weight_ih_l1 = torch.nn.Parameter(self.rnn.weight_ih_l1.detach().clone())
            new_agent.policy_decoder = torch.nn.Parameter(self.policy_decoder.detach().clone())
            new_agent.value_decoder = torch.nn.Parameter(self.value_decoder.detach().clone())
            new_agent.input_encoder = torch.nn.Parameter(self.input_encoder.detach().clone())
            new_agent.policy_decoder_bias = torch.nn.Parameter(self.policy_decoder_bias.detach().clone())
            new_agent.value_decoder_bias = torch.nn.Parameter(self.value_decoder_bias.detach().clone())
            new_agent.input_encoder_bias = torch.nn.Parameter(self.input_encoder_bias.detach().clone())

        return new_agent

    def instantiate(self):
        new_agent = RNN_Agent(num_nodes=36, num_layers=2, channels=1, spatial=5, kernel=None, device="cpu")

        new_agent.rnn.bias_hh_l0 = torch.nn.Parameter(self.rnn.bias_hh_l0.clone())
        new_agent.rnn.bias_hh_l1 = torch.nn.Parameter(self.rnn.bias_hh_l1.clone())
        new_agent.rnn.bias_ih_l0 = torch.nn.Parameter(self.rnn.bias_hh_l0.clone())
        new_agent.rnn.bias_ih_l1 = torch.nn.Parameter(self.rnn.bias_hh_l1.clone())
        new_agent.rnn.weight_hh_l0 = torch.nn.Parameter(self.rnn.weight_hh_l0.clone())
        new_agent.rnn.weight_hh_l1 = torch.nn.Parameter(self.rnn.weight_hh_l1.clone())
        new_agent.rnn.weight_ih_l0 = torch.nn.Parameter(self.rnn.weight_ih_l0.clone())
        new_agent.rnn.weight_ih_l1 = torch.nn.Parameter(self.rnn.weight_ih_l1.clone())
        new_agent.policy_decoder = self.policy_decoder.clone()
        new_agent.value_decoder = self.value_decoder.clone()
        new_agent.input_encoder = self.input_encoder.clone()
        new_agent.policy_decoder_bias = self.policy_decoder_bias.clone()
        new_agent.value_decoder_bias = self.value_decoder_bias.clone()
        new_agent.input_encoder_bias = self.input_encoder_bias.clone()
        new_agent.epsilon = self.epsilon
        new_agent.id = self.id
        return new_agent
    #
    # def pretrain_agent_input(self, epochs, obs_data, use_channels=True):
    #     batch_size = 250
    #     channels = self.input_channels
    #     decoders = torch.nn.Linear(in_features=self.spatial * self.input_channels, out_features=self.input_size)
    #     optim = torch.optim.Adam([self.input_encoder] + list(decoders.parameters()), lr=.0001)
    #
    #     loss_hist = []
    #
    #     for i in range(epochs):
    #         optim.zero_grad()
    #         batch = torch.from_numpy(obs_data[np.random.choice(np.arange(len(obs_data)), size=batch_size, replace=False)]).float()
    #         encoded = torch.sigmoid(batch @ self.input_encoder)
    #         encoded = encoded.view((batch_size, channels * self.spatial))
    #         decoded = decoders(encoded)
    #         loss = torch.sqrt(torch.mean(torch.pow(batch - decoded, 2)))
    #         reg = .001 * torch.sum(torch.abs(self.input_encoder))
    #         loss_hist.append(loss.detach().cpu().item())
    #         print("epoch", i, "loss", loss_hist[-1], "reg", reg.detach().cpu().item())
    #         loss = loss + reg
    #         loss.backward()
    #         optim.step()

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

# train(model, 20, "cpu", 0.001)