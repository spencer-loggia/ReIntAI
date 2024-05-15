from agent.agents import WaterworldAgent, DisjointWaterWorldAgent, FCWaterworldAgent
from agent.evolve import EvoController
import pickle
import torch
# torch.set_default_dtype(torch.float64)
import sys

if __name__=="__main__":
    if sys.platform == "linux":
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
    FC_AGENT = True
    DISJOINT_CRITIC = False
    BP_THROUGH_TIME = True # FC only rn
    EPOCHS = 30000
    EPSILON_START = .75
    EPSILON_DECAY = 4000
    NUM_BASE = 5
    ALGORITM = "a3c"
    LOG_MAX_LR = -2.5
    LOG_MIN_LR = -3.5

    if FC_AGENT:
        if DISJOINT_CRITIC:
            path = "models/dijoint_fc_evo_32_bias"
            agent = [FCWaterworldAgent(num_nodes=4, spatial=32, channels=3, input_channels=2, device="cpu", decode_node=None, through_time=BP_THROUGH_TIME) for _ in range(NUM_BASE)]
            # with open("models/wworld_fc_pretrain7.pkl", "rb") as f:
            #     in_enc = pickle.load(f)
            # agent.input_encoder = in_enc
        else:
            path = "models/fc_evo_32_bias_no_back"
            agent = [FCWaterworldAgent(num_nodes=4, spatial=32, channels=3, input_channels=2, through_time=BP_THROUGH_TIME, device="cpu") for _ in range(NUM_BASE)]
    else:
        if DISJOINT_CRITIC:
            path = "models/disjoint_evo_7"
            agent = [DisjointWaterWorldAgent(num_nodes=4, spatial=7, channels=3, input_channels=2, device="cpu") for _ in range(NUM_BASE)]
            with open("models/wworld_val_pretrain7.pkl", "rb") as f:
                agent.value_decoder = pickle.load(f)
        else:
            path = "models/evo_7"
            agent = [WaterworldAgent(num_nodes=4, spatial=7, channels=3, input_channels=2, device="cpu") for _ in range(NUM_BASE)]

        with open("models/wworld_pretrain7.pkl", "rb") as f:
            in_enc = pickle.load(f)
        agent.input_encoder = in_enc.to(torch.float64)

    evo = EvoController(seed_agent=agent, epochs=EPOCHS, num_base=NUM_BASE, num_workers=6,
                        min_agents=1, max_agents=3, min_gen=1, max_gen=1, log_min_lr=LOG_MIN_LR, log_max_lr=LOG_MAX_LR,
                        algo=ALGORITM, start_epsilon=EPSILON_START, inverse_eps_decay=EPSILON_DECAY, worker_device="cpu", viz=False)
    # evo.load_model("/Users/loggiasr/Projects/ReIntAI/models/dijoint_fc_evo_32_bias/snap_500_5.64_.pkl")
    evo.controller(mp=True, fbase=path)