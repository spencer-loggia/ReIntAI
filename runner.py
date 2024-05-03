from agent.agents import WaterworldAgent, DisjointWaterWorldAgent, FCWaterworldAgent
from agent.evolve import EvoController
import pickle

if __name__=="__main__":
    FC_AGENT = True
    DISJOINT_CRITIC = False
    EPOCHS = 20000
    EPSILON_START = .95
    EPSILON_DECAY = 4000
    ALGORITM = "a3c"
    LOG_MAX_LR = -3.5
    LOG_MIN_LR = -8

    if FC_AGENT:
        path = "models/fc_evo_32_bias"
        agent = FCWaterworldAgent(num_nodes=4, spatial=32, channels=3, input_channels=2, device="cpu")
        # with open("models/wworld_fc_pretrain7.pkl", "rb") as f:
        #     in_enc = pickle.load(f)
        # agent.input_encoder = in_enc
    else:
        if DISJOINT_CRITIC:
            path = "models/disjoint_evo_7"
            agent = DisjointWaterWorldAgent(num_nodes=4, spatial=7, channels=3, input_channels=2, device="cpu")
            with open("models/wworld_val_pretrain7.pkl", "rb") as f:
                agent.value_decoder = pickle.load(f)
        else:
            path = "models/evo_7"
            agent = WaterworldAgent(num_nodes=4, spatial=7, channels=3, input_channels=2, device="cpu")

        with open("models/wworld_pretrain7.pkl", "rb") as f:
            in_enc = pickle.load(f)
        agent.input_encoder = in_enc

    evo = EvoController(seed_agent=agent, epochs=EPOCHS, num_base=4, num_workers=10,
                        min_agents=1, max_agents=3, min_gen=1, max_gen=1, log_min_lr=LOG_MIN_LR, log_max_lr=LOG_MAX_LR,
                        algo=ALGORITM, start_epsilon=EPSILON_START, inverse_eps_decay=EPSILON_DECAY, worker_device="cpu")
    # evo.load_model("/Users/loggiasr/Projects/ReIntAI/models/fc_evo_32/snap_4500_5.52_.pkl")
    evo.controller(mp=True, fbase=path)