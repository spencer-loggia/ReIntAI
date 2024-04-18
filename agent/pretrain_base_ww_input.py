import sys

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


    all_observations = []
    for i in range(gens):
        env = waterworld_v4.parallel_env(n_pursuers=4, n_coop=1, n_sensors=n_sensors,
                                                 max_cycles=50, speed_features=False, pursuer_max_accel=.5,
                                                 encounter_reward=0.0)
        observations, infos = env.reset()

        while env.agents:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actions)
            for agent in env.agents:
                all_observations.append(observations[agent])
        env.close()

    observations = np.stack(all_observations)[np.random.choice(np.arange(len(all_observations)), size=len(all_observations), replace=False)]
    np.savetxt(out_file, observations)
