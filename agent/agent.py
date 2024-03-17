import torch
from intrinsic.model import Intrinsic
from pettingzoo.sisl import waterworld_v4


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
        self.channels = channels
        self.input_size = sensors * 5 + 2
        self.core_model = Intrinsic(num_nodes, node_shape=(1, channels, spatial, spatial), kernel_size=kernel)
        input_encoder = torch.empty((self.input_size, self.spatial * self.spatial * self.channels))
        self.input_encoder = torch.nn.Parameter(torch.nn.init.xavier_normal_(input_encoder))

        action_decoder = torch.empty(self.spatial ** 2, action_dim)
        self.action_decoder = torch.nn.Parameter(torch.nn.init.xavier_normal_(action_decoder))

    def forward(self, X):
        """
        :param X: Agent Sensor Data
        :return:
        """
        with torch.no_grad():
            encoded_input = (X @ self.input_encoder).view(self.channels, self.spatial, self.spatial)
            in_states = torch.zeros_like(self.core_model.states)
            in_states[0, :, :, :] += encoded_input
        out_states = self.core_model(in_states)
        action = out_states[2, 0, :, :].flatten() @ self.action_decoder
        return action


class Evolve():
    def __init__(self, num_agents, num_sensors=20):
        self.env_constructor = waterworld_v4
        self.num_agents = num_agents
        self.sensors = num_sensors
        self.agents = [WaterworldAgent(num_nodes=4, sensors=num_sensors) for _ in range(num_agents)]

    def play(self, human_interface=True):
        if human_interface:
            env = self.env_constructor.env(render_mode="human", n_pursuers=self.num_agents, n_coop=1, n_sensors=self.sensors, speed_features=False)
        else:
            env = self.env_constructor.env(n_pursuers=self.num_agents, n_coop=1, n_sensors=self.sensors, speed_features=False)
        env.reset()
        agent_dict = {agent_name: self.agents[i] for i, agent_name in enumerate(env.agents)}
        for i, agent in enumerate(env.agent_iter()):
            observation, reward, termination, truncation, info = env.last()
            observation = torch.from_numpy(observation)
            if termination or truncation:
                action = None
            else:
                # this is where you would insert your policy
                action = agent_dict[agent](observation)
                #action = env.action_space(agent).sample()
            env.step(action.detach().cpu())
        env.close()


if __name__=="__main__":
    evo = Evolve(2)
    evo.play()
