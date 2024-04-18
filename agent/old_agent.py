import copy
import math
import pickle
import random
import sys
import timeit
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import multiprocessing as mp

import intrinsic.util
from intrinsic.model import Intrinsic
from intrinsic.util import triu_to_square
from pettingzoo.sisl import waterworld_v4
from scipy.ndimage import gaussian_filter


def compute_ac_error(true_reward, value_est: List[torch.Tensor], action_likelihood, action_entropy, gamma):
    """

    :param true_reward: Reward from taking action a_t in state s_t
    :param value_est: Estimated average future value of being in state s_t
    :param action_likelihood: The likelihoods of the sampled actions taken given the policy distribution.
    :param action_entropy: The entropy of the action distributions.
    :param gamma: temporal discount factor.
    :return:
    """
    device = value_est[0].device
    true_reward = [0.] + true_reward  # the expected reward of taking no action in step 0. is 0.
    R = torch.Tensor([0.], device=device)
    val_loss = torch.Tensor([0.], device=device)
    policy_loss = torch.Tensor([0.], device=device)
    value_est.reverse()
    action_likelihood.reverse()
    for i, instant_reward in enumerate(reversed(true_reward)):
        # the current estimate of future reward at time T - i
        R = R.detach() * gamma + instant_reward
        # compute temporal difference
        td = R - value_est[i]  # episode future reward in s_t - expected future reward in s_t
        val_loss = val_loss + torch.pow(td, 2)  # critic network estimates expected state values over all action dist.
        policy_loss = policy_loss - action_likelihood[i] * (td.detach())  # local td is the advantage of the action selected over state value.
    return val_loss, policy_loss, torch.sum(torch.Tensor(action_entropy, device=device))


class WaterworldAgent():
    """
    """
    def __init__(self, input_channels=2, num_nodes=4, channels=3, spatial=5, kernel=3, sensors=20, action_dim=2, device="cpu", *args, **kwargs):
        """
        Defines the core agents with extended modules for input and output.
        input node is always 0, reward signal node is always 1, output node is always 2
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.spatial = spatial
        self.action_dim = action_dim
        self.channels = channels
        self.device = device
        self.num_sensors = sensors
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
        self.v_lost_hist = []
        self.p_loss_hist = []

    def clone(self, fuzzy=True):
        new_agent = WaterworldAgent(num_nodes=self.core_model.num_nodes,
                                    channels=self.channels, spatial=self.spatial,
                                    kernel=self.core_model.edge.kernel_size, sensors=self.num_sensors,
                                    action_dim=self.action_dim,
                                    device=self.device,
                                    input_channels=self.input_channels)
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
        return new_agent

    def detach(self):
        self.core_model.detach(reset_intrinsic=True)
        self.value_decoder.detach()
        self.policy_decoder.detach()
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
        agent_heads = [self.value_decoder, self.policy_decoder] + self.core_model.parameters()
        return agent_heads

    def __call__(self, x=None):
        return self.forward(x)

    def forward(self, X):
        """
        :param X: Agent Sensor Data
        :return: Mu, Sigma, Value - the mean and variance of the action distribution, and the state value estimate
        """
        # create a state matrix for injection into core from input observation
        with torch.no_grad():
            encoded_input = (X @ self.input_encoder).view(self.input_channels, self.spatial, self.spatial)
            in_states = torch.zeros_like(self.core_model.states)
            in_states[0, :self.input_channels, :, :] += encoded_input
        # run a model time step
        out_states = self.core_model(in_states)
        # compute next action and value estimates
        action_params = out_states[2, 0, :, :].flatten() @ self.policy_decoder
        value_est = out_states[1, 0, :, :].flatten() @ self.value_decoder
        angle_mu = action_params[0] * 2 * torch.pi + .001
        angle_sigma = torch.abs(action_params[1]) + .001
        rad_conc1 = torch.abs(action_params[2]) + .001
        rad_conc2 = torch.abs(action_params[3]) + .001
        return angle_mu, angle_sigma, rad_conc1, rad_conc2, value_est


class Evolve():
    def __init__(self, start_agent:WaterworldAgent, n_base_agents=2, instance_per_base=2, num_sensors=20, device="cpu"):
        self.num_agents = n_base_agents * instance_per_base
        self.num_base = n_base_agents
        self.instances = instance_per_base
        self.lr = .0001
        self.sensors = num_sensors
        self.device = device
        self.base_agent = [start_agent.clone(fuzzy=True) for _ in range(n_base_agents)]
        self.critic_optimizers = [torch.optim.Adam(self.base_agent[i].core_model.parameters() +
                                                   [self.base_agent[i].value_decoder], lr=self.lr)
                                  for i in range(n_base_agents)]
        self.policy_optimizers = [torch.optim.Adam(self.base_agent[i].core_model.parameters() +
                                                   [self.base_agent[i].policy_decoder], lr=self.lr)
                                  for i in range(n_base_agents)]
        self.v_loss_hist = []
        self.p_loss_hist = []

    def play(self, human_interface=False, cyles=200):
        agents = [[self.base_agent[i].detach()] for i in range(self.num_base)]
        scores = [0.] * self.num_base
        for j in range(self.num_base):
            for i in range(self.instances - 1):
                agents[j].append(self.base_agent[j].instantiate())
        if human_interface:
            env = waterworld_v4.parallel_env(render_mode="human", n_pursuers=self.num_agents, n_coop=1,
                                                    n_sensors=self.sensors, max_cycles=cyles, speed_features=False,
                                                    pursuer_max_accel=.5, encounter_reward=0.0)
        else:
            env = waterworld_v4.parallel_env(n_pursuers=self.num_agents, n_coop=1, n_sensors=self.sensors,
                                                    max_cycles=cyles, speed_features=False, pursuer_max_accel=.5,
                                                    encounter_reward=0.0)
        env.reset()
        agent_dict = {}
        for i, base in enumerate(agents):
            for j, agent in enumerate(base):
                agent_name = env.agents[i * len(base) + j]
                agent_dict[agent_name] = {"base_index": i,
                                           "model": agents[i][j],
                                           "action_likelihood": [],
                                           "entropy": [],
                                           "inst_r": [],
                                           "value": [],
                                           "terminate": False,
                                           "failure": False}

        observations, infos = env.reset()

        while env.agents:
            # this is where you would insert your policy
            actions = {}
            for agent in env.agents:
                mu, sigma, alpha, beta, v_hat = agent_dict[agent]["model"](torch.from_numpy(observations[agent]))
                with torch.no_grad():
                    if torch.isnan(mu + sigma + alpha + v_hat).any():
                        agent_dict[agent]["failure"] = True
                        agent_dict[agent]["terminate"] = True
                if not agent_dict[agent]["terminate"]:
                    adist = torch.distributions.VonMises(loc=mu, concentration=sigma)
                    rdist = torch.distributions.Beta(alpha, beta)
                    action_rad = adist.sample()
                    if action_rad is None:
                        print("WARN: Von Mises sampling stage failed. Concentrations were", alpha, beta,
                              ". Acting randomly...")
                        action_rad = torch.rand((1,), device=self.device) * torch.pi * 2
                    action_mag = rdist.sample()
                    action = torch.stack([action_mag * torch.cos(action_rad), action_mag * torch.sin(action_rad)])
                    # print("ACTION", action)
                    likelihood = adist.log_prob(action_rad)
                    likelihood = likelihood + rdist.log_prob(action_mag)
                    entropy = rdist.entropy()
                    agent_dict[agent]["action_likelihood"].append(likelihood)
                    agent_dict[agent]["entropy"].append(entropy)
                    agent_dict[agent]["value"].append(v_hat.clone())
                    actions[agent] = action.detach().cpu().numpy()
                else:
                    actions[agent] = np.array([0., 0.])
            observations, rewards, terminations, truncations, infos = env.step(actions)
            for i, agent in enumerate(env.agents):
                base = agent_dict[agent]["base_index"]
                if terminations[agent]:
                    agent_dict[agent]["terminate"] = True
                if not agent_dict[agent]["terminate"]:
                    reward = rewards[agent]
                    agent_dict[agent]["inst_r"].append(reward)
                    scores[base] += float(reward)
                else:
                    # penalty for reaching divergent state
                    scores[base] -= 10000.

        env.close()
        return agent_dict, scores

    def plot_loss(self):
        try:
            fig, axs = plt.subplots(3)
            axs[0].plot(gaussian_filter(np.array(self.v_loss_hist), 20))
            axs[1].plot(gaussian_filter(np.array(self.p_loss_hist), 10))
            fig.show()
        except Exception:
            print("showing history failed")
            pass

    def evolve(self, generations=2000, disp_iter=100):
        for i in range(self.num_base):
            self.policy_optimizers[i].zero_grad()
            self.critic_optimizers[i].zero_grad()
        e_hist = []
        for i, gen in enumerate(range(generations)):
            print("Generation", i)
            h_int = False
            if ((gen + 1) % disp_iter) == 0:
                h_int = True
            gen_info, base_scores = self.play(h_int)
            best_base = base_scores.index(max(base_scores))
            total_loss = torch.Tensor([0.], device=self.device)
            for agent in gen_info.keys():
                agent_info = gen_info[agent]
                if agent_info["base_index"] != best_base:
                    continue
                val_loss, policy_loss, entropy_loss = compute_ac_error(agent_info["inst_r"],
                                                                       agent_info["value"],
                                                                       agent_info["action_likelihood"],
                                                                       agent_info["entropy"],
                                                                       gamma=.95)
                print("VL:", val_loss.detach().cpu().item(), "PL", policy_loss.detach().cpu().item(), "H:", entropy_loss.detach().cpu().item())
                self.v_loss_hist.append(val_loss.detach().cpu().item())
                self.p_loss_hist.append(policy_loss.detach().cpu().item())
                e_hist.append(entropy_loss.detach().cpu().item())
                a = 1.0
                b = .5
                c = -0.2
                total_loss = total_loss + a * val_loss + b * policy_loss
            try:
                total_loss.backward()
                self.critic_optimizers[best_base].step()
                self.policy_optimizers[best_base].step()
            except RuntimeError:
                print("Optimization step failure on gen", gen)
                continue

            # propogate best agent
            best_agent = self.base_agent[best_base]
            # rand mutate only 1
            fuzzy = [False] * self.num_base
            fuzzy[random.randint(0, self.num_base - 1)] = True
            for i, ba in enumerate(self.base_agent):
                if i != best_base:
                    self.base_agent[i] = best_agent.clone(fuzzy=fuzzy[i])
                    self.optimizers[i].param_groups[0] = torch.optim.Adam(self.base_agent[i].parameters(),
                                                                          lr=self.lr).param_groups[0]


if __name__ == "__main__":
    try:
        load_f = sys.argv[1]
        with open(load_f, "rb") as f:
            evo = pickle.load(f)
            evo.instances = 2
            evo.num_agents = 2
            evo.plot_loss()
            evo.play(human_interface=True)
    except IndexError:
        evo = Evolve(1, 6)
    evo.optimizers = [torch.optim.Adam(evo.base_agent[i].parameters(), lr=evo.lr) for i in range(evo.num_base)]
    #evo.evolve(generations=2000)
    with open("/Users/loggiasr/Projects/ReIntAI/models/wworld_1.pkl", "wb") as f:
        pickle.dump(evo, f)
