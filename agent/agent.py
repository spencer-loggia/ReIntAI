import copy
import pickle
import timeit

import matplotlib.pyplot as plt
import torch
from torch import multiprocessing as mp
from intrinsic.model import Intrinsic
from intrinsic.util import triu_to_square
from pettingzoo.sisl import waterworld_v4

def compute_ac_error(true_reward, value_est, action_likelihood, action_entropy, gamma):
    """

    :param true_reward:
    :param value_est:
    :param action_likelihood:
    :param action_entropy:
    :param gamma:
    :return:
    """
    R = torch.Tensor([0.])
    val_loss = torch.Tensor([0.])
    policy_loss = torch.Tensor([0.])
    value_est.reverse()
    action_likelihood.reverse()
    for i, instant_reward in enumerate(reversed(true_reward + [0.])):
        R = R * gamma + instant_reward
        val_loss = val_loss + (R - value_est[i]) ** 2
        policy_loss = policy_loss + -1 * action_likelihood[i] * (R - value_est[i])
    return torch.sqrt(val_loss), policy_loss, torch.sum(torch.Tensor(action_entropy))


class WaterworldAgent():
    """

    """
    def __init__(self, num_nodes=4, channels=3, spatial=5, kernel=3, sensors=20, action_dim=2, *args, **kwargs):
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
        self.num_sensors = sensors
        self.input_size = sensors * 5 + 2
        self.core_model = Intrinsic(num_nodes, node_shape=(1, channels, spatial, spatial), kernel_size=kernel)

        # transform inputs to core model space
        input_encoder = torch.empty((self.input_size, self.spatial * self.spatial * self.channels))
        self.input_encoder = torch.nn.Parameter(torch.nn.init.xavier_normal_(input_encoder))

        # produce logits for mean and covariance of policy distribution
        policy_decoder = torch.empty(self.spatial ** 2, 4)
        self.policy_decoder = torch.nn.Parameter(torch.nn.init.xavier_normal_(policy_decoder))

        # produce critic state value estimates
        value_decoder = torch.empty(self.spatial ** 2, 1)
        self.value_decoder = torch.nn.Parameter(torch.nn.init.xavier_normal_(value_decoder))
        self.v_lost_hist = []
        self.p_loss_hist = []

    def instantiate(self):
        new_agent = WaterworldAgent(num_nodes=self.core_model.num_nodes,
                                    channels=self.channels, spatial=self.spatial,
                                    kernel=self.core_model.edge.kernel_size, sensors=self.num_sensors,
                                    action_dim=self.action_dim)
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

    def parameters(self):
        agent_heads = [self.value_decoder, self.policy_decoder, self.input_encoder] + self.core_model.parameters()
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
            encoded_input = (X @ self.input_encoder).view(self.channels, self.spatial, self.spatial)
            in_states = torch.zeros_like(self.core_model.states)
            in_states[0, :, :, :] += encoded_input
        # run a model time step
        out_states = self.core_model(in_states)
        # compute next action and value estimates
        action_params = out_states[2, 0, :, :].flatten() @ self.policy_decoder
        value_est = out_states[1, 0, :, :].flatten() @ self.value_decoder
        angle_mu = action_params[0] * 2 * torch.pi
        angle_sigma = torch.abs(action_params[1])
        rad_conc1 = torch.abs(action_params[2])
        rad_conc2 = torch.abs(action_params[3])
        return angle_mu, angle_sigma, rad_conc1, rad_conc2, value_est


class Evolve():
    def __init__(self, num_agents, num_sensors=20):
        self.env_constructor = waterworld_v4
        self.num_agents = num_agents
        self.sensors = num_sensors
        self.base_agent = WaterworldAgent(num_nodes=5, sensors=num_sensors)
        self.optimizers = torch.optim.Adam(self.base_agent.parameters(), lr=.00001)

    def play(self, human_interface=False):
        self.base_agent.detach()
        agents = [self.base_agent]
        for i in range(self.num_agents - 1):
            agents.append(self.base_agent.instantiate())
        if human_interface:
            env = self.env_constructor.parallel_env(render_mode="human", n_pursuers=self.num_agents, n_coop=1,
                                                    n_sensors=self.sensors, max_cycles=150, speed_features=False,
                                                    pursuer_max_accel=.5, encounter_reward=0.0)
        else:
            env = self.env_constructor.parallel_env(n_pursuers=self.num_agents, n_coop=1, n_sensors=self.sensors,
                                                    max_cycles=150, speed_features=False, pursuer_max_accel=.5,
                                                    encounter_reward=0.0)
        env.reset()
        agent_dict = {agent_name: {"model": agents[i],
                                   "action_likelihood": [],
                                   "entropy": [],
                                   "inst_r": [],
                                   "value": []} for i, agent_name in enumerate(env.agents)}

        observations, infos = env.reset()

        while env.agents:
            # this is where you would insert your policy
            actions = {}
            terminations = None
            for agent in env.agents:
                mu, sigma, alpha, beta, v_hat = agent_dict[agent]["model"](torch.from_numpy(observations[agent]))
                adist = torch.distributions.VonMises(loc=mu, concentration=sigma)
                rdist = torch.distributions.Beta(alpha, beta)
                action_rad = adist.sample()
                if action_rad is None:
                    print("WARN: Von Mises sampling stage failed. Concentrations were", alpha, beta,
                          ". Acting randomly...")
                    action_rad = torch.rand((1,)) * torch.pi * 2
                action_mag = rdist.sample()
                action = torch.Tensor([action_mag * torch.cos(action_rad), action_mag * torch.sin(action_rad)])
                # print("ACTION", action)
                likelihood = adist.log_prob(action_rad)
                likelihood = likelihood + rdist.log_prob(action_mag)
                entropy = rdist.entropy()
                agent_dict[agent]["action_likelihood"].append(likelihood)
                agent_dict[agent]["entropy"].append(entropy)
                agent_dict[agent]["value"].append(v_hat.clone())
                actions[agent] = action
            observations, rewards, terminations, truncations, infos = env.step(actions)
            for i, agent in enumerate(env.agents):
                agent_dict[agent]["inst_r"].append(rewards[agent])

        env.close()
        return agent_dict

    def evolve(self, generations=1000, disp_iter=50):
        self.optimizers.zero_grad()
        self.v_loss_hist = []
        self.p_loss_hist = []
        e_hist = []
        for i, gen in enumerate(range(generations)):
            print("Generation", i)
            h_int = False
            if ((gen + 1) % disp_iter) == 0:
                h_int = True
            gen_info = self.play(h_int)
            total_loss = torch.Tensor([0.])
            for agent in gen_info.keys():
                agent_info = gen_info[agent]
                val_loss, policy_loss, entropy_loss = compute_ac_error(agent_info["inst_r"],
                                                                       agent_info["value"],
                                                                       agent_info["action_likelihood"],
                                                                       agent_info["entropy"],
                                                                       gamma=.95)
                print("VL:", val_loss.detach().cpu().item(), "PL", policy_loss.detach().cpu().item(), "H:", entropy_loss.detach().cpu().item())
                self.v_loss_hist.append(val_loss)
                self.p_loss_hist.append(policy_loss)
                e_hist.append(entropy_loss)
                a = 1.0
                b = .5
                c = -0.2
                total_loss = total_loss + a * val_loss + b * policy_loss
            try:
                total_loss.backward()
                self.optimizers.step()
            except RuntimeError:
                print("Optimization step failure on gen", gen)
                continue
        try:
            fig, axs = plt.subplots(3)
            axs[0].plot(self.v_loss_hist)
            axs[1].plot(self.p_loss_hist)
            axs[2].plot(e_hist)
            fig.show()
        except Exception:
            pass


if __name__=="__main__":
    evo = Evolve(4)
    evo.evolve()
    with open("/Users/loggiasr/Projects/ReIntAI/models/wworld_1.pkl", "wb") as f:
        pickle.dump(evo, f)
