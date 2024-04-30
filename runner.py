from agent.agents import WaterworldAgent, DisjointWaterWorldAgent
from agent.evolve import EvoController
import pickle

if __name__=="__main__":
    DISJOINT_CRITIC = True
    EPOCHS = 20000
    EPSILON_START = .95
    EPSILON_DECAY = 5000
    ALGORITM = "a3c"
    LOG_MAX_LR = -7
    LOG_MIN_LR = -13

    if DISJOINT_CRITIC:
        path = "models/disjoint_evo_7"
        agent = DisjointWaterWorldAgent(num_nodes=4, spatial=7, channels=3, input_channels=3)
        with open("models/wworld_val_pretrain7.pkl", "rb") as f:
            agent.value_decoder = pickle.load(f)
    else:
        path = "models/evo_7"
        agent = WaterworldAgent(num_nodes=4, spatial=7, channels=3, input_channels=3)

    with open("models/wworld_pretrain7.pkl", "rb") as f:
        in_enc = pickle.load(f)
    agent.input_encoder = in_enc

    evo = EvoController(seed_agent=agent, epochs=EPOCHS, num_base=4, num_workers=10,
                        min_agents=1, max_agents=3, min_gen=1, max_gen=1, log_min_lr=LOG_MIN_LR, log_max_lr=LOG_MAX_LR,
                        algo=ALGORITM, start_epsilon=EPSILON_START, inverse_eps_decay=EPSILON_DECAY)
    evo.load_model("/Users/loggiasr/Projects/ReIntAI/models/evo_7/snap_2500_4.71_.pkl")
    evo.controller(mp=True, fbase=path)