import torch
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
    for i, instant_reward in enumerate(reversed(true_reward)):
        R = R * gamma + instant_reward
        val_loss = val_loss + (R - value_est[i]) ** 2
        policy_loss = policy_loss + -1 * action_likelihood[i] * (R - value_est[i])
    return val_loss, policy_loss, torch.sum(torch.Tensor(action_entropy))


class WaterworldAgent(torch.nn.Module):
    """

    """
    def __init__(self, num_nodes=4, channels=3, spatial=7, kernel=3, sensors=20, action_dim=2, *args, **kwargs):
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
        self.input_size = sensors * 5 + 2
        self.core_model = Intrinsic(num_nodes, node_shape=(1, channels, spatial, spatial), kernel_size=kernel)

        # transform inputs to core model space
        input_encoder = torch.empty((self.input_size, self.spatial * self.spatial * self.channels))
        self.input_encoder = torch.nn.Parameter(torch.nn.init.xavier_normal_(input_encoder))

        # produce logits for mean and covariance of policy distribution
        cov_dim = int(action_dim * (action_dim + 1) / 2)
        policy_decoder = torch.empty(self.spatial ** 2, action_dim + cov_dim)
        self.policy_decoder = torch.nn.Parameter(torch.nn.init.xavier_normal_(policy_decoder))

        # produce critic state value estimates
        value_decoder = torch.empty(self.spatial ** 2, 1)
        self.value_decoder = torch.nn.Parameter(torch.nn.init.xavier_normal_(value_decoder))

    def detach(self):
        self.core_model.detach(reset_intrinsic=True)
        self.value_decoder.detach()
        self.policy_decoder.detach()

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
        action_mu = torch.tanh(action_params[:self.action_dim])
        action_sigma = action_params[self.action_dim:]
        cov = torch.sigmoid(triu_to_square(action_sigma, n=self.action_dim, includes_diag=True))
        cov = cov @ cov.T  # ensure is positive definite
        return action_mu, cov, value_est


class Evolve():
    def __init__(self, num_agents, num_sensors=20):
        self.env_constructor = waterworld_v4
        self.num_agents = num_agents
        self.sensors = num_sensors
        self.agents = [WaterworldAgent(num_nodes=5, sensors=num_sensors) for _ in range(num_agents)]
        self.optimizers = [torch.optim.Adam(a.parameters(), lr=.0001) for a in self.agents]

    def play(self, human_interface=True):
        if human_interface:
            env = self.env_constructor.env(render_mode="human", n_pursuers=self.num_agents, n_coop=1, n_sensors=self.sensors, max_cycles=150, speed_features=False)
        else:
            env = self.env_constructor.env(n_pursuers=self.num_agents, n_coop=1, n_sensors=self.sensors, max_cycles=150, speed_features=False)
        env.reset()
        agent_dict = {agent_name: {"model": self.agents[i],
                                   "optim": self.optimizers[i],
                                   "action_likelihood": [],
                                   "entropy": [],
                                   "inst_r": [],
                                   "value": []} for i, agent_name in enumerate(env.agents)}
        for i in range(4):
            try:
                for i, agent in enumerate(env.agent_iter()):
                    observation, reward, termination, truncation, info = env.last()
                    observation = torch.from_numpy(observation)
                    if termination or truncation:
                        action = None
                    else:
                        # this is where you would insert your policy
                        mu, sigma, v_hat = agent_dict[agent]["model"](observation)
                        # construct action distribution
                        try:
                            adist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma)
                        except ValueError:
                            continue
                        action = adist.sample()
                        likelihood = adist.log_prob(action)
                        entropy = adist.entropy()
                        agent_dict[agent]["action_likelihood"].append(likelihood.clone())
                        agent_dict[agent]["entropy"].append(entropy.clone())
                        agent_dict[agent]["inst_r"].append(float(reward))
                        agent_dict[agent]["value"].append(v_hat.clone())
                    if action is None:
                        env.step(None)
                    else:
                        env.step(action.detach().cpu())
                break
            except AssertionError:
                pass
        env.close()
        return agent_dict

    def evolve(self, generations=10000, disp_iter=100):
        [optim.zero_grad() for optim in self.optimizers]
        for i, gen in enumerate(range(generations)):
            print("Generation", i)
            [a.detach() for a in self.agents]
            h_int = False
            if (gen % disp_iter) == 0:
                h_int = True
            gen_info = self.play(h_int)
            for agent in gen_info.keys():
                agent_info = gen_info[agent]
                optim = agent_info["optim"]
                val_loss, policy_loss, entropy_loss = compute_ac_error(agent_info["inst_r"],
                                                                       agent_info["value"],
                                                                       agent_info["action_likelihood"],
                                                                       agent_info["entropy"],
                                                                       gamma=.95)
                a = b = c = 1.
                total_loss = a * val_loss + b * policy_loss + c * entropy_loss
                try:
                    total_loss.backward()
                    optim.step()
                except RuntimeError:
                    continue
        pass


if __name__=="__main__":
    evo = Evolve(1)
    evo.evolve()
