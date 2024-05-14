from agent.agents import WaterworldAgent, DisjointWaterWorldAgent, FCWaterworldAgent
from agent.evolve import EvoController
from RNNTrain import RNN_Agent
import pickle
import torch
torch.set_default_dtype(torch.float64)
import sys

if __name__=="__main__":
    if sys.platform == "linux":
        import torch.multiprocessing as mp
        mp.set_start_method('forkserver', force=True)
    FC_AGENT = True
    DISJOINT_CRITIC = False
    RNN_AGENT = False
    EPOCHS = 20000
    EPSILON_START = 1.0
    EPSILON_DECAY = 4000
    ALGORITM = "a3c"
    LOG_MAX_LR = -3
    LOG_MIN_LR = -8

    if FC_AGENT:
        if DISJOINT_CRITIC:
            path = "models/dijoint_fc_evo_32_bias"
            agent = FCWaterworldAgent(num_nodes=4, spatial=32, channels=3, input_channels=2, device="cpu", decode_node=None)
            # with open("models/wworld_fc_pretrain7.pkl", "rb") as f:
            #     in_enc = pickle.load(f)
            # agent.input_encoder = in_enc
        else:
            path = "models/fc_evo_32_bias_no_back"
            agent = FCWaterworldAgent(num_nodes=4, spatial=32, channels=3, input_channels=2, device="cpu")
    elif RNN_AGENT:
        path = "models/rnn"
        agent = RNN_Agent(num_nodes=36, num_layers=2, device="cpu")
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
        agent.input_encoder = in_enc.to(torch.float64)

    evo = EvoController(seed_agent=agent, epochs=EPOCHS, num_base=5, num_workers=2,
                        min_agents=1, max_agents=3, min_gen=1, max_gen=1, log_min_lr=LOG_MIN_LR, log_max_lr=LOG_MAX_LR,
                        algo=ALGORITM, start_epsilon=EPSILON_START, inverse_eps_decay=EPSILON_DECAY, worker_device="cpu")
    # evo.load_model("/Users/loggiasr/Projects/ReIntAI/models/evo_7/snap_8500_13.59_.pkl")
    evo.controller(mp=True, fbase=path)