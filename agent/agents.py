import copy
import math
import pickle
import random
import sys
import timeit
from typing import List
import randomname

import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import torch
from torch import multiprocessing as mp

import intrinsic.util
from intrinsic.model import Intrinsic
from intrinsic.util import triu_to_square
from pettingzoo.sisl import waterworld_v4


class WaterworldAgent:
    def __init__(self, input_channels=2, num_nodes=4, channels=3, spatial=5, kernel=3, sensors=20, action_dim=2,
                 device="cpu", *args, **kwargs):
        """
        Defines the core agents with extended modules for input and output.
        input node is always 0, reward signal node is always 1, output node is always 2
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.id = randomname.get_name()
        self.generation = 0.
        self.fitness = 0.  # running tally of past generation scores decaying with .8
        self.spatial = spatial
        self.action_dim = action_dim
        self.channels = channels
        self.device = device
        self.num_sensors = sensors
        self.version = 0
        self.input_channels = input_channels
        self.input_size = sensors * 5 + 2
        self.core_model = Intrinsic(num_nodes, node_shape=(1, channels, spatial, spatial), kernel_size=kernel)

        # transform inputs to core model space
        input_encoder = torch.empty((self.input_size, self.spatial * self.spatial * input_channels), device=device)
        input_encoder = torch.nn.init.xavier_normal_(input_encoder)
        self.input_encoder = torch.nn.Parameter(input_encoder)

        # produce logits for mean and covariance of policy distribution
        policy_decoder = torch.empty((self.spatial ** 2, 4), device=device)
        self.policy_decoder = torch.nn.Parameter(torch.nn.init.xavier_normal_(policy_decoder))

        # produce critic state value estimates
        value_decoder = torch.empty((self.spatial ** 2, 1), device=device)
        self.value_decoder = torch.nn.Parameter(torch.nn.init.xavier_normal_(value_decoder))
        self.v_loss = None
        self.p_loss = None

    def clone(self, fuzzy=True):
        new_agent = WaterworldAgent(num_nodes=self.core_model.num_nodes,
                                    channels=self.channels, spatial=self.spatial,
                                    kernel=self.core_model.edge.kernel_size, sensors=self.num_sensors,
                                    action_dim=self.action_dim,
                                    device=self.device, input_channels=self.input_channels)
        if not fuzzy:
            new_agent.id = self.id
            new_agent.version = self.version
            new_agent.fitness = self.fitness
            new_agent.v_loss = self.v_loss
            new_agent.p_loss = self.p_loss

        with torch.no_grad():
            new_core = self.core_model.clone(fuzzy=fuzzy)
            new_agent.core_model = new_core
            new_agent.policy_decoder = torch.nn.Parameter(self.policy_decoder.detach().clone())
            new_agent.value_decoder = torch.nn.Parameter(self.value_decoder.detach().clone())
            new_agent.input_encoder = torch.nn.Parameter(self.input_encoder.detach().clone())
        return new_agent

    def instantiate(self):
        new_agent = WaterworldAgent(num_nodes=self.core_model.num_nodes,
                                    channels=self.channels, spatial=self.spatial,
                                    kernel=self.core_model.edge.kernel_size, sensors=self.num_sensors,
                                    action_dim=self.action_dim,
                                    device=self.device, input_channels=self.input_channels)
        new_core = self.core_model.instantiate()
        new_agent.core_model = new_core
        new_agent.policy_decoder = self.policy_decoder.clone()
        new_agent.value_decoder = self.value_decoder.clone()
        new_agent.input_encoder = self.input_encoder.clone()
        new_agent.id = self.id
        return new_agent

    def detach(self):
        self.core_model.detach(reset_intrinsic=True)
        self.value_decoder.detach()
        self.policy_decoder.detach()
        self.input_encoder.detach()
        return self

    def pretrain_agent_input(self, epochs, obs_data, use_channels=True):
        spatial = self.spatial
        batch_size = 250
        kernel, pad = intrinsic.util.conv_identity_params(spatial, 4)
        channels = self.input_channels
        out_feature = self.input_size
        decoders = torch.nn.Sequential(torch.nn.Conv2d(in_channels=channels, out_channels=channels * 2, kernel_size=kernel, padding=pad),
                                      torch.nn.MaxPool2d(2),
                                      torch.nn.Conv2d(in_channels=channels * 2, out_channels=channels * 4, kernel_size=kernel, padding=pad),
                                      torch.nn.MaxPool2d(2),
                                      torch.nn.Flatten(),
                                      torch.nn.Linear(in_features=(math.floor(spatial / 4) ** 2) * channels * 4,
                                                      out_features=out_feature))
        optim = torch.optim.Adam([self.input_encoder] + list(decoders.parameters()), lr=.0001)

        loss_hist = []

        for i in range(epochs):
            optim.zero_grad()
            batch = torch.from_numpy(obs_data[np.random.choice(np.arange(len(obs_data)), size=batch_size, replace=False)]).float()
            encoded = torch.sigmoid(batch @ self.input_encoder)
            encoded = encoded.view((batch_size, channels, spatial, spatial))
            decoded = decoders(encoded)
            loss = torch.sqrt(torch.mean(torch.pow(batch - decoded, 2)))
            loss_hist.append(loss.detach().cpu().item())
            print("epoch", i, "loss", loss_hist[-1])
            loss.backward()
            optim.step()

        plt.plot(loss_hist)
        plt.show()

    def parameters(self):
        agent_heads = [self.input_encoder, self.value_decoder, self.policy_decoder] + self.core_model.parameters()
        return agent_heads

    def __eq__(self, other):
        return hash(self)

    def __hash__(self):
        return hash(self.id)

    def set_grad(self, grad):
        # MUST BE IN SAME ORDER AS PARAMETERS (BAD PRACTICE BUT LAZY)
        self.value_decoder.grad = grad[1]
        self.policy_decoder.grad = grad[2]
        self.input_encoder.grad = grad[0]
        self.core_model.set_grad(grad[3:])

    def __call__(self, x=None):
        return self.forward(x)

    def forward(self, X):
        """
        :param X: Agent Sensor Data
        :return: Mu, Sigma, Value - the mean and variance of the action distribution, and the state value estimate
        """
        # create a state matrix for injection into core from input observation
        encoded_input = (X @ self.input_encoder).view(self.input_channels, self.spatial, self.spatial)
        in_states = torch.zeros_like(self.core_model.states)
        # in_states[0, :self.input_channels, :, :] = .25 * in_states[0, :self.input_channels, :, :] + .75 * encoded_input
        in_states[0, :self.input_channels, :, :] = encoded_input
        # run a model time step
        for i in range(1):
            out_states = self.core_model(in_states)
        # compute next action and value estimates
        action_params = out_states[2, 0, :, :].flatten() @ self.policy_decoder
        value_est = out_states[1, 0, :, :].flatten() @ self.value_decoder
        angle_mu = action_params[0] * 2 * torch.pi + .001
        angle_sigma = torch.abs(action_params[1]) + .001
        rad_conc1 = torch.abs(action_params[2]) + .001
        rad_conc2 = torch.abs(action_params[3]) + .001
        return angle_mu, angle_sigma, rad_conc1, rad_conc2, value_est

