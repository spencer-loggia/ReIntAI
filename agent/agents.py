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
from intrinsic.model import Intrinsic, FCIntrinsic
from intrinsic.util import triu_to_square
from pettingzoo.sisl import waterworld_v4


class WaterworldAgent:
    def __init__(self, input_channels=2, num_nodes=4, channels=3, spatial=5, kernel=3, sensors=20, action_dim=2, epsilon=0,
                 device="cpu", *args, **kwargs):
        """
        Defines the core agents with extended modules for input and output.
        input node is always 0, reward signal node is always 1, output node is always 2
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.id = randomname.get_name()
        self.debug = False
        self.generation = 0.
        self.fitness = 0.  # running tally of past generation scores decaying with .8
        self.spatial = spatial
        self.action_dim = action_dim
        self.epsilon = epsilon
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

        self.value_decoder_bias = torch.nn.Parameter(torch.zeros((1,), device=self.device) + .001)

        self.policy_decoder_bias = torch.nn.Parameter(torch.zeros((1,), device=self.device) + .001)

        self.input_encoder_bias = torch.nn.Parameter(torch.zeros((1,), device=self.device) + .001)

        self.v_loss = None
        self.p_loss = None

    def clone(self, fuzzy=True, set_dev=None):
        new_agent = type(self)(num_nodes=self.core_model.num_nodes,
                                    channels=self.channels, spatial=self.spatial,
                                    kernel=self.core_model.edge.kernel_size, sensors=self.num_sensors,
                                    action_dim=self.action_dim,
                                    device=self.device, input_channels=self.input_channels)
        if set_dev is None:
            set_dev = self.device
        if not fuzzy:
            new_agent.id = self.id
            new_agent.version = self.version
            new_agent.fitness = self.fitness
            new_agent.v_loss = self.v_loss
            new_agent.p_loss = self.p_loss
        new_agent.epsilon = self.epsilon

        with torch.no_grad():
            new_core = self.core_model.clone(fuzzy=fuzzy, device=set_dev)
            new_agent.core_model = new_core
            new_agent.policy_decoder = torch.nn.Parameter(self.policy_decoder.detach().to(set_dev))
            new_agent.value_decoder = torch.nn.Parameter(self.value_decoder.detach().to(set_dev))
            new_agent.input_encoder = torch.nn.Parameter(self.input_encoder.detach().to(set_dev))
            new_agent.policy_decoder_bias = torch.nn.Parameter(self.policy_decoder_bias.detach().to(set_dev))
            new_agent.value_decoder_bias = torch.nn.Parameter(self.value_decoder_bias.detach().to(set_dev))
            new_agent.input_encoder_bias = torch.nn.Parameter(self.input_encoder_bias.detach().to(set_dev))
        return new_agent

    def instantiate(self):
        new_agent = type(self)(num_nodes=self.core_model.num_nodes,
                                    channels=self.channels, spatial=self.spatial,
                                    kernel=self.core_model.edge.kernel_size, sensors=self.num_sensors,
                                    action_dim=self.action_dim,
                                    device=self.device, input_channels=self.input_channels)
        new_core = self.core_model.instantiate()
        new_agent.core_model = new_core
        new_agent.policy_decoder = self.policy_decoder.clone()
        new_agent.value_decoder = self.value_decoder.clone()
        new_agent.input_encoder = self.input_encoder.clone()
        new_agent.policy_decoder_bias = self.policy_decoder_bias.clone()
        new_agent.value_decoder_bias = self.value_decoder_bias.clone()
        new_agent.input_encoder_bias = self.input_encoder_bias.clone()
        new_agent.epsilon = self.epsilon
        new_agent.id = self.id
        return new_agent

    def detach(self):
        self.core_model.detach(reset_intrinsic=True)
        self.value_decoder.detach()
        self.policy_decoder.detach()
        self.input_encoder.detach()
        self.value_decoder_bias.detach()
        self.policy_decoder_bias.detach()
        self.input_encoder_bias.detach()
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
            reg = .001 * torch.sum(torch.abs(self.input_encoder))
            loss_hist.append(loss.detach().cpu().item())
            print("epoch", i, "loss", loss_hist[-1], "reg", reg.detach().cpu().item())
            loss = loss + reg
            loss.backward()
            optim.step()

        plt.plot(loss_hist)
        plt.show()

    def parameters(self):
        agent_heads = [self.input_encoder, self.input_encoder_bias, self.value_decoder,
                       self.value_decoder_bias, self.policy_decoder, self.policy_decoder_bias] + self.core_model.parameters()
        return agent_heads

    def set_grad(self, grad):
        # MUST BE IN SAME ORDER AS PARAMETERS (BAD PRACTICE BUT LAZY)
        self.policy_decoder_bias.grad = grad[5]
        self.policy_decoder.grad = grad[4]

        self.value_decoder_bias.grad = grad[3]
        self.value_decoder.grad = grad[2]

        self.input_encoder_bias.grad = grad[1]
        self.input_encoder.grad = grad[0]

        self.core_model.set_grad(grad[6:])

    def __eq__(self, other):
        return hash(self)

    def __hash__(self):
        return hash(self.id)

    def __call__(self, x=None, *args, **kwargs):
        return self.forward(x)

    def forward(self, X, r=None):
        """
        :param X: Agent Sensor Data
        :param r: instant reward form last state
        :return: Mu, Sigma, Value - the mean and variance of the action distribution, and the state value estimate
        """
        # create a state matrix for injection into core from input observation
        encoded_input = ((X + self.input_encoder_bias) @ self.input_encoder).view(self.input_channels, self.spatial, self.spatial)
        in_states = torch.zeros_like(self.core_model.states)
        mask = in_states.bool()
        mask[0, :self.input_channels, :, :] = True
        # in_states[0, :self.input_channels, :, :] = .25 * in_states[0, :self.input_channels, :, :] + .75 * encoded_input
        in_states[0, :self.input_channels, :, :] = encoded_input
        if r is not None:
            in_states[2, 0, :, :] = r
        # run a model time step
        for i in range(1):
            out_states = self.core_model(in_states, mask)
        # compute next action and value estimates
        action_params = (out_states[1, 0, :, :].flatten() + self.policy_decoder_bias) @ self.policy_decoder
        value_est = (out_states[2, 0, :, :].flatten() + self.value_decoder_bias) @ self.value_decoder
        c1 = torch.pow(action_params[0:2], 2) + 1.0
        c2 = torch.pow(action_params[2:], 2) + 1.0
        return c1, c2, value_est


class DisjointWaterWorldAgent(WaterworldAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_decoder = torch.empty((self.input_size, 1), device=self.device)
        self.value_decoder = torch.nn.Parameter(torch.nn.init.xavier_normal_(self.value_decoder))

    def forward(self, X, r=None):
        """
        :param X: Agent Sensor Data
        :param r: instant reward form last state
        :return: Mu, Sigma, Value - the mean and variance of the action distribution, and the state value estimate
        """
        # create a state matrix for injection into core from input observation
        encoded_input = ((X + self.input_encoder_bias) @ self.input_encoder).view(self.input_channels, self.spatial, self.spatial)
        in_states = torch.zeros_like(self.core_model.states)
        mask = in_states.bool()
        mask[0, :self.input_channels, :, :] = True
        # in_states[0, :self.input_channels, :, :] = .25 * in_states[0, :self.input_channels, :, :] + .75 * encoded_input
        in_states[0, :self.input_channels, :, :] = encoded_input
        if r is not None:
            in_states[2, 0, :, :] = r
        # run a model time step
        for i in range(1):
            out_states = self.core_model(in_states, mask)
        # compute next action and value estimates
        action_params = (out_states[1, 0, :, :].flatten() + self.policy_decoder_bias) @ self.policy_decoder
        value_est = ((X + self.value_decoder_bias) @ self.value_decoder).flatten(0) # out_states[2, 0, :, :].flatten() @ self.value_decoder
        c1 = torch.exp(action_params[0:2]) + .1
        c2 = torch.exp(action_params[2:]) + .1
        return c1, c2, value_est


class FCWaterworldAgent(WaterworldAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

        self.core_model = FCIntrinsic(num_nodes=4, node_shape=(1, self.channels, self.spatial), device=self.device)
        self.kernel_size = None

    def parameters(self):
        agent_heads = [self.input_encoder, self.input_encoder_bias, self.value_decoder,
                       self.value_decoder_bias, self.policy_decoder, self.policy_decoder_bias] + self.core_model.parameters()
        return agent_heads

    def set_grad(self, grad):
        # MUST BE IN SAME ORDER AS PARAMETERS (BAD PRACTICE BUT LAZY)
        self.policy_decoder_bias.grad = grad[5]
        self.policy_decoder.grad = grad[4]

        self.value_decoder_bias.grad = grad[3]
        self.value_decoder.grad = grad[2]

        self.input_encoder_bias.grad = grad[1]
        self.input_encoder.grad = grad[0]

        self.core_model.set_grad(grad[6:])

    def forward(self, X, r=None):
        """
        :param X: Agent Sensor Data
        :param r: instant reward form last state
        :return: Mu, Sigma, Value - the mean and variance of the action distribution, and the state value estimate
        """
        X = X.to(self.device)
        # create a state matrix for injection into core from input observation
        encoded_input = ((X + self.input_encoder_bias) @ self.input_encoder).view(self.spatial, self.input_channels).transpose(0, 1)
        in_states = torch.zeros_like(self.core_model.states)
        mask = in_states.bool()
        mask[0, :self.input_channels, :] = True
        # in_states[0, :self.input_channels, :, :] = .25 * in_states[0, :self.input_channels, :, :] + .75 * encoded_input
        in_states[0, :self.input_channels, :] = encoded_input
        if r is not None:
            in_states[3, 0, :] = r
        # run a model time step
        for i in range(1):
            out_states = self.core_model(in_states, mask)
        # compute next action and value estimates
        action_params = out_states[1, 0, :].flatten() @ self.policy_decoder + self.policy_decoder_bias
        value_est = (out_states[2, 0, :].flatten() @ self.value_decoder).flatten() + self.value_decoder_bias # out_states[2, 0, :, :].flatten() @ self.value_decoder
        if self.debug:
            value_est.register_hook(lambda grad: print("Grad Val Est", torch.abs(grad).sum()))
        c1 = torch.relu(action_params[0:2]) + 1.0
        c2 = torch.relu(action_params[2:]) + 1.0
        return c1, c2, value_est

    def clone(self, fuzzy=True):
        new_agent = FCWaterworldAgent(num_nodes=self.core_model.num_nodes,
                                    channels=self.channels, spatial=self.spatial, sensors=self.num_sensors,
                                    action_dim=self.action_dim,
                                    device=self.device, input_channels=self.input_channels)
        if not fuzzy:
            new_agent.id = self.id
            new_agent.version = self.version
            new_agent.fitness = self.fitness
            new_agent.v_loss = self.v_loss
            new_agent.p_loss = self.p_loss
        new_agent.epsilon = self.epsilon

        with torch.no_grad():
            new_core = self.core_model.clone(fuzzy=fuzzy)
            new_agent.core_model = new_core
            new_agent.policy_decoder = torch.nn.Parameter(self.policy_decoder.detach().clone())
            new_agent.value_decoder = torch.nn.Parameter(self.value_decoder.detach().clone())
            new_agent.input_encoder = torch.nn.Parameter(self.input_encoder.detach().clone())
            new_agent.policy_decoder_bias = torch.nn.Parameter(self.policy_decoder_bias.detach().clone())
            new_agent.value_decoder_bias = torch.nn.Parameter(self.value_decoder_bias.detach().clone())
            new_agent.input_encoder_bias = torch.nn.Parameter(self.input_encoder_bias.detach().clone())
        return new_agent

    def instantiate(self):
        new_agent = FCWaterworldAgent(num_nodes=self.core_model.num_nodes,
                                    channels=self.channels, spatial=self.spatial, sensors=self.num_sensors,
                                    action_dim=self.action_dim,
                                    device=self.device, input_channels=self.input_channels)
        new_core = self.core_model.instantiate()
        new_agent.core_model = new_core
        new_agent.policy_decoder = self.policy_decoder.clone()
        new_agent.value_decoder = self.value_decoder.clone()
        new_agent.input_encoder = self.input_encoder.clone()
        new_agent.policy_decoder_bias = self.policy_decoder_bias.clone()
        new_agent.value_decoder_bias = self.value_decoder_bias.clone()
        new_agent.input_encoder_bias = self.input_encoder_bias.clone()
        new_agent.epsilon = self.epsilon
        new_agent.id = self.id
        return new_agent

    def pretrain_agent_input(self, epochs, obs_data, use_channels=True):
        batch_size = 250
        channels = self.input_channels
        decoders = torch.nn.Linear(in_features=self.spatial * self.input_channels, out_features=self.input_size)
        optim = torch.optim.Adam([self.input_encoder] + list(decoders.parameters()), lr=.0001)

        loss_hist = []

        for i in range(epochs):
            optim.zero_grad()
            batch = torch.from_numpy(obs_data[np.random.choice(np.arange(len(obs_data)), size=batch_size, replace=False)]).float()
            encoded = torch.sigmoid(batch @ self.input_encoder)
            encoded = encoded.view((batch_size, channels * self.spatial))
            decoded = decoders(encoded)
            loss = torch.sqrt(torch.mean(torch.pow(batch - decoded, 2)))
            reg = .001 * torch.sum(torch.abs(self.input_encoder))
            loss_hist.append(loss.detach().cpu().item())
            print("epoch", i, "loss", loss_hist[-1], "reg", reg.detach().cpu().item())
            loss = loss + reg
            loss.backward()
            optim.step()

        plt.plot(loss_hist)
        plt.show()

