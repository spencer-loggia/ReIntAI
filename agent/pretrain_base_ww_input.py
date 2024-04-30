import os
import sys
from reward_functions import return_from_reward

import numpy as np
import torch
from pettingzoo.sisl import waterworld_v4

"""
Create a large database of agent observations for pretraining
"""

if __name__=="__main__":
    n_sensors = int(sys.argv[1])
    gens = int(sys.argv[2])
    out_file = sys.argv[3]

    all_returns = []
    all_observations = []

    for i in range(gens):
        print("Running gen", i)
        env = waterworld_v4.parallel_env(n_pursuers=3, n_coop=1, n_sensors=n_sensors,
                                                 max_cycles=600, speed_features=False, pursuer_max_accel=.5,
                                                 encounter_reward=0.0)
        observations, infos = env.reset()
        local_rewards = {a:[] for a in env.agents}
        local_observations = {a:[] for a in env.agents}
        while env.agents:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actions)
            for agent in env.agents:
                local_observations[agent].append(observations[agent])
                local_rewards[agent].append(rewards[agent])
        for agent in local_rewards.keys():
            if len(local_rewards[agent]) < 15:
                continue
            all_observations.append(np.array(local_observations[agent][:-15]))
            all_returns.append(return_from_reward(torch.Tensor(local_rewards[agent][:-15]), .97).detach().cpu().numpy())
            print("avg. return:", np.mean(all_returns[-1]))
        env.close()

    observations = np.concatenate(all_observations)
    shuffle_indexes = np.random.choice(np.arange(len(observations)), size=len(observations), replace=False).astype(int)
    observations = observations[shuffle_indexes]
    returns = np.concatenate(all_returns)[shuffle_indexes]
    returns = returns - returns.mean()
    returns = returns / returns.std()
    np.savetxt(os.path.join(out_file, "obs_data.txt"), observations)
    np.savetxt(os.path.join(out_file, "returns.txt"), returns)
