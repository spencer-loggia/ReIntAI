from agent.agents import WaterworldAgent
from agent.evolve import EvoController
import pickle

if __name__=="__main__":
    agent = WaterworldAgent(num_nodes=4, spatial=7, channels=3, input_channels=3)
    with open("/Users/loggiasr/Projects/ReIntAI/models/wworld_pretrain7.pkl", "rb") as f:
        in_enc = pickle.load(f)
    agent.input_encoder = in_enc
    base = pickle
    evo = EvoController(seed_agent=agent, epochs=20000, num_base=4, num_workers=10,
                        min_agents=1, max_agents=3, min_gen=1, max_gen=1, log_min_lr=-7, log_max_lr=-3)
    # evo.load_model("/Users/loggiasr/Projects/ReIntAI/models/evo_7/snap_2999_6.91_.pkl")
    evo.controller(mp=True, fbase="/Users/loggiasr/Projects/ReIntAI/models/evo_7")