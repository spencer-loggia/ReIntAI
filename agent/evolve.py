import copy
import math
import os.path
import pickle
import random
import sys
import time
import timeit
from typing import List
import networkx

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import randomname

matplotlib.use('Qt5Agg')

import numpy as np
import torch
from torch.multiprocessing import Queue, Process

import intrinsic.util
from intrinsic.model import Intrinsic
from intrinsic.util import triu_to_square
from agent.agents import WaterworldAgent
from pettingzoo.sisl import waterworld_v4
from scipy.ndimage import uniform_filter1d

def compute_ac_error(true_reward, value_est: List[torch.Tensor], action_likelihood, action_entropy, gamma):
    """

    :param true_reward: Reward from taking action a_t in state s_t
    :param value_est: Estimated average future value of being in state s_t
    :param action_likelihood: The likelihoods of the sampled actions taken given the policy distribution.
    :param action_entropy: The entropy of the action distributions.
    :param gamma: temporal discount factor.
    :return:
    """
    device = value_est[0].device
    true_reward = [0.] + true_reward  # the expected reward of taking no action in step 0. is 0.
    R = torch.Tensor([0.], device=device)
    val_loss = None
    policy_loss = None
    # val_loss = torch.Tensor([], device=device)
    # policy_loss = torch.Tensor([], device=device)
    value_est.reverse()
    action_likelihood.reverse()
    for i, instant_reward in enumerate(reversed(true_reward)):
        # the current estimate of future reward at time T - i
        R = R.detach() * gamma + instant_reward
        # compute temporal difference
        td = R - value_est[i]  # episode future reward in s_t - expected future reward in s_t
        val_loss_i = torch.pow(td, 2)
        policy_loss_i = -action_likelihood[i] * td.detach()
        if val_loss is None:
            val_loss = val_loss_i
            policy_loss = policy_loss_i
        else:
            val_loss = val_loss + val_loss_i  # critic network estimates expected state values over all action dist.
            policy_loss = policy_loss + policy_loss_i  # local td is the advantage of the action selected over state value.
    return val_loss, policy_loss, torch.sum(torch.Tensor(action_entropy, device=device))


def episode(base_agents, copies, min_cycles=600, max_cycles=600, sensors=20, human=False, device="cpu"):
    num_base = len(base_agents)
    agents = [[base_agents[i].detach()] for i in range(num_base)]
    scores = [0.] * num_base
    cycles = random.randint(min_cycles, max_cycles)
    num_agents = num_base
    for j in range(num_base):
        for i in range(copies[j] - 1):
            num_agents += 1
            agents[j].append(base_agents[j].instantiate())
    if human:
        env = waterworld_v4.parallel_env(render_mode="human", n_pursuers=num_agents, n_coop=1,
                                         n_sensors=sensors, max_cycles=cycles, speed_features=False,
                                         pursuer_max_accel=1.0, encounter_reward=0., food_reward=5.0,
                                         poison_reward=-5.0, thrust_penalty=-.05)
    else:
        env = waterworld_v4.parallel_env(n_pursuers=num_agents, n_coop=1, n_sensors=sensors,
                                         max_cycles=cycles, speed_features=False, pursuer_max_accel=.5,
                                         encounter_reward=0., food_reward=5.0, poison_reward=-5.0,
                                         thrust_penalty=-.05)
    env.reset()
    agent_dict = {}
    env_agent_index = 0
    for i, base in enumerate(agents):
        for j, agent in enumerate(base):
            agent_name = env.agents[env_agent_index]
            agent_dict[agent_name] = {"base_index": i,
                                      "base_name": agent.id,
                                      "model": agents[i][j],
                                      "action_likelihood": [],
                                      "entropy": [],
                                      "inst_r": [],
                                      "value": [],
                                      "terminate": False,
                                      "failure": False}
            env_agent_index += 1

    observations, infos = env.reset()

    while env.agents:
        # this is where you would insert your policy
        actions = {}
        for agent in env.agents:
            mu, sigma, alpha, beta, v_hat = agent_dict[agent]["model"](torch.from_numpy(observations[agent]))
            with torch.no_grad():
                if torch.isnan(mu + sigma + alpha + v_hat).any():
                    agent_dict[agent]["failure"] = True
                    agent_dict[agent]["terminate"] = True
            if not agent_dict[agent]["terminate"]:
                adist = torch.distributions.VonMises(loc=mu, concentration=sigma)
                rdist = torch.distributions.Beta(alpha, beta)
                action_rad = adist.sample()
                if action_rad is None:
                    print("WARN: Von Mises sampling stage failed. Concentrations were", alpha, beta,
                          ". Acting randomly...")
                    action_rad = torch.rand((1,), device=device) * torch.pi * 2
                action_mag = rdist.sample()
                action = torch.stack([action_mag * torch.cos(action_rad), action_mag * torch.sin(action_rad)])
                # print("ACTION", action)
                likelihood = adist.log_prob(action_rad)
                likelihood = likelihood + rdist.log_prob(action_mag)
                entropy = .2 * rdist.entropy() + adist.entropy()
                # if entropy > 2000:
                #     from matplotlib import pyplot as plt
                #     with torch.no_grad():
                #         x = 2 * torch.pi * torch.arange(1000) / 1000
                #         l = torch.stack([adist.log_prob(xi) for xi in x])
                #     plt.ylim((-100, 0.1))
                #     plt.plot(x.numpy(), l.numpy())
                #     plt.show()
                agent_dict[agent]["action_likelihood"].append(likelihood)
                agent_dict[agent]["entropy"].append(entropy)
                agent_dict[agent]["value"].append(v_hat.clone())
                actions[agent] = action.detach().cpu().numpy()
            else:
                actions[agent] = np.array([0., 0.])
        try:
            observations, rewards, terminations, truncations, infos = env.step(actions)
        except ValueError:
            for agent in env.agents:
                agent_dict[agent]["failure"] = True
                agent_dict[agent]["terminate"] = True
                base = agent_dict[agent]["base_index"]
                scores[base] = None
        for i, agent in enumerate(env.agents):
            base = agent_dict[agent]["base_index"]
            if not agent_dict[agent]["terminate"]:
                reward = rewards[agent]
                agent_dict[agent]["inst_r"].append(reward)
                scores[base] += float(reward)
                if terminations[agent]:
                    scores[base] = None
                    agent_dict[agent]["terminate"] = True
    scores = [s / cycles for s in scores]
    env.close()
    return agent_dict, scores


def local_evolve(q, generations, base_agents, copies, value_lr=1e-9, policy_lr=1e-9, device="cpu"):
    num_base = len(base_agents)
    value_optimizers = [torch.optim.Adam(params=a.core_model.parameters() + [a.value_decoder], lr=value_lr)
                        for a in base_agents]
    policy_optimizers = [torch.optim.Adam(params=a.core_model.parameters() + [a.policy_decoder], lr=policy_lr)
                        for a in base_agents]
    #if value_optim_params is not None:
        #for i in range(num_base):
            #value_optimizers[i].param_groups[0] = value_optim_params[i]
            #policy_optimizers[i].param_groups[0] = policy_optim_params[i]

    fail_tracker = [False for _ in range(num_base)]
    stat_tracker = {a.id: {"gradient": [0. for _ in a.parameters()],
                           "value_loss": [],
                           "policy_loss": [],
                           "entropy": [],
                           "fitness": [],
                           "copies": copies[i] * generations,
                           "failure": False} for i, a in enumerate(base_agents)}

    # TODO: Track fitness in stat tracker. Update model fitness with average fitness and aver losses from worker (since no update is applied here)

    for gen in range(generations):
        for i in range(num_base):
            value_optimizers[i].zero_grad()
            policy_optimizers[i].zero_grad()
        h_int = False
        total_loss = [torch.Tensor([0.], device=device) for _ in range(num_base)]
        gen_info, base_scores = episode(base_agents, copies, h_int)
        for a in stat_tracker.keys():
            stat_tracker[a]["value_loss"].append([])
            stat_tracker[a]["policy_loss"].append([])
            stat_tracker[a]["entropy"].append([])

        for agent in gen_info.keys():
            agent_info = gen_info[agent]
            if not agent_info["failure"] and not stat_tracker[agent_info["base_name"]]["failure"]:
                val_loss, policy_loss, entropy_loss = compute_ac_error(agent_info["inst_r"],
                                                                       agent_info["value"],
                                                                       agent_info["action_likelihood"],
                                                                       agent_info["entropy"],
                                                                       gamma=.95)
                if torch.isnan(val_loss + policy_loss):
                    print("NaN is gradient!", agent_info["base_name"])
                stat_tracker[agent_info["base_name"]]["value_loss"][-1].append(val_loss.detach().cpu().item())
                stat_tracker[agent_info["base_name"]]["entropy"][-1].append(entropy_loss.detach().cpu().item())
                stat_tracker[agent_info["base_name"]]["policy_loss"][-1].append(policy_loss.detach().cpu().item())
                stat_tracker[agent_info["base_name"]]["copies"] += 1
                a = .05
                b = .01
                c = .001
                total_loss[agent_info["base_index"]] = total_loss[agent_info["base_index"]] + a * val_loss + b * policy_loss - c * entropy_loss
            else:
                stat_tracker[agent_info["base_name"]]["value_loss"][-1].append(None)
                stat_tracker[agent_info["base_name"]]["policy_loss"][-1].append(None)
                stat_tracker[agent_info["base_name"]]["entropy"][-1].append(None)
                stat_tracker[agent_info["base_name"]]["failure"] = True
                fail_tracker[agent_info["base_index"]] = True
        for j, score in enumerate(base_scores):
            a = base_agents[j]
            if stat_tracker[a.id]["fitness"] is None:
                continue
            if score is None:
                stat_tracker[a.id]["fitness"] = None
            else:
                stat_tracker[a.id]["fitness"].append(score)
        try:
            for i in range(num_base):
                if not fail_tracker[i]:
                    a = base_agents[i]
                    total_loss[i].backward()
                    for j, p in enumerate(a.parameters()):
                        if not torch.isnan(p.grad).any():
                            stat_tracker[a.id]["gradient"][j] += p.grad
                        else:
                            print("NaN grad", a.id)
                            stat_tracker[a.id]["failure"] = True
                            stat_tracker[a.id]["gradient"][j] += torch.zeros_like(p.grad)
                            fail_tracker[i] = True
                    # value_optimizers[i].step()
                    # policy_optimizers[i].step()
        except RuntimeError:
            print("Optimization step failure on gen", gen)
            continue
    # average and cast to numpy
    for k in stat_tracker.keys():
        for pid in range(len(stat_tracker[k]["gradient"])):
            stat_tracker[k]["gradient"][pid] *= 1e-2
        stat_tracker[k]["value_loss"] = np.nanmean(np.array(stat_tracker[k]["value_loss"], dtype=float))
        stat_tracker[k]["policy_loss"] = np.nanmean(np.array(stat_tracker[k]["policy_loss"], dtype=float))
        stat_tracker[k]["entropy"] = np.nanmean(np.array(stat_tracker[k]["entropy"], dtype=float))
        if stat_tracker[k]["fitness"] is not None:
            stat_tracker[k]["fitness"] = np.mean(stat_tracker[k]["fitness"])

    q.put(([a.detach() for a in base_agents], stat_tracker))
    return


def _compute_loss_values(arr, copies=None, window=5):
    len_hist = len(arr)
    start = min(len_hist, window)
    arr = np.array(arr[-start:], dtype=float)
    if copies is None:
        score = np.nansum(arr)
    else:
        copies = np.array(copies[-start:], dtype=float)
        copies = copies / np.sum(copies)
        score = np.nansum(arr * copies)
    return score


class EvoController:
    def __init__(self, seed_agent: WaterworldAgent, epochs=10, num_base=4,
                 min_gen=10, max_gen=30, min_agents=3, max_agents=8,
                 log_min_lr=-13, log_max_lr=-8, num_workers=6, worker_device="cpu"):
        self.num_base = num_base
        self.log_min_lr = log_min_lr
        self.log_max_lr = log_max_lr
        self.epochs = epochs
        self.min_gen = min_gen
        self.max_gen = max_gen
        self.min_agents = min_agents
        self.max_agents = max_agents
        self.last_grad = [0. for _ in seed_agent.parameters()]

        self.num_workers = num_workers
        self.sensors = seed_agent.num_sensors
        self.base_agent = [seed_agent.clone(fuzzy=False)]
        self.value_opitmizers = {}
        self.policy_opitmizers = {}
        self.integration_q = Queue(maxsize=100)
        self.worker_device = worker_device
        self.num_integrations = 0
        self.evo_tree = networkx.DiGraph()
        self.evo_tree.add_node(seed_agent.id, fitness=[], vloss=[], ploss=[], copies=[], entropy=[])
        self.value_loss_hist = []
        self.policy_loss_hist = []
        self.fitness_hist = []

        # local display figure
        self.fig, self.axs = plt.subplots(3)
        self.axs[0].set_ylabel("Value loss")
        self.axs[1].set_ylabel("Policy loss")

        # global loss display figures
        #self.global_fig, self.global_axs = plt.subplots(3)
        self.axs[2].set_ylabel("Fitness")

        # self.global_fig.suptitle("Overall Loss Progress")

        # tree display figure
        # self.tree_fig, self.tree_axs = plt.subplots(1)
        # self.tree_fig.suptitle("Evolutionary Tree")

    def spawn_worker(self, mp=True):
        for i in range(len(self.base_agent)):
            self._add_optimizer_set(self.base_agent[i])
        num_gens = random.randint(self.min_gen, self.max_gen)
        num_agents = random.randint(self.min_agents, self.max_agents)
        select_base_idx = np.random.choice(np.arange(len(self.base_agent)), size=num_agents)
        use_base_idx, copies = np.unique(select_base_idx, return_counts=True)
        use_base = []
        for i in use_base_idx:
            use_base.append(self.base_agent[i].clone(fuzzy=False))
        # use_base = [self.base_agent[i].clone(fuzzy=False) for i in use_base_idx]
        # v_optim = [self.value_opitmizers[use_base[i].id].param_groups[0] for i in range(len(use_base))]
        # p_optim = [self.policy_opitmizers[use_base[i].id].param_groups[0] for i in range(len(use_base))]
        print("OPTIM:", num_gens, "generations,", num_agents, "agents of types:", [a.id for a in use_base])
        if mp:
            p = Process(target=local_evolve,
                        args=(self.integration_q, num_gens, use_base, copies.tolist()))
            return p
        else:
            local_evolve(self.integration_q, num_gens, use_base, copies.tolist())
            return None

    def multiclone(self, agent1: WaterworldAgent, agent2: WaterworldAgent, equal=False):
        new_agent = WaterworldAgent(num_nodes=agent1.core_model.num_nodes,
                                    channels=agent1.channels, spatial=agent1.spatial,
                                    kernel=agent1.core_model.edge.kernel_size, sensors=agent1.num_sensors,
                                    action_dim=agent1.action_dim,
                                    device=agent1.device, input_channels=agent1.input_channels)
        with torch.no_grad():
            if equal:
                new_core_1 = agent1.core_model.clone(fuzzy=False)
                lincomb = .5
            else:
                new_core_1 = agent1.core_model.clone(fuzzy=False)
                lincomb = random.random() * 2
            new_core_1.edge.chan_map = torch.nn.Parameter((1 - lincomb) * new_core_1.edge.chan_map.detach() +
                                                          lincomb * agent2.core_model.edge.chan_map.detach())
            new_core_1.edge.init_weight = torch.nn.Parameter((1 - lincomb) * new_core_1.edge.init_weight.detach() +
                                                             lincomb * agent2.core_model.edge.init_weight.detach())
            new_core_1.edge.plasticity = torch.nn.Parameter((1 - lincomb) * new_core_1.edge.plasticity.detach() +
                                                            lincomb * agent2.core_model.edge.plasticity.detach())
            new_core_1.resistance = torch.nn.Parameter((1 - lincomb) * new_core_1.resistance.detach() +
                                                            lincomb * agent2.core_model.resistance.detach())

            new_agent.core_model = new_core_1

            new_agent.policy_decoder = torch.nn.Parameter((1 - lincomb) * agent1.policy_decoder.detach().clone()
                                                          + (lincomb) * agent2.policy_decoder.detach().clone())
            new_agent.value_decoder = torch.nn.Parameter((1 - lincomb) * agent1.value_decoder.detach().clone()
                                                         + (lincomb) * agent2.value_decoder.detach().clone())
            new_agent.input_encoder = torch.nn.Parameter((1 - lincomb) * agent1.input_encoder.detach().clone()
                                                         + (lincomb) * agent2.input_encoder.detach().clone())
        new_agent.id = randomname.get_name()
        new_agent.version = 0
        return new_agent

    def survival(self):
        # select the most fit in the overall pool.
        # all_agents = [a.clone(fuzzy=False) for a in self.base_agent]
        kill_prob = 1 / (2 * self.num_workers)
        if random.random() > kill_prob:
            return
        num_survivors = min(self.num_base - 1, math.ceil(self.num_base * .75))
        if num_survivors == 0:
            num_survivors = 1
        print("Agent pool")

        def _val(a: WaterworldAgent):
            aid = a.id
            version = a.version
            all_ = self.evo_tree.nodes[aid]["fitness"]
            all_v = self.evo_tree.nodes[aid]["vloss"]
            all_p = self.evo_tree.nodes[aid]["ploss"]
            all_h = self.evo_tree.nodes[aid]["entropy"]
            fit = _compute_loss_values(all_, self.evo_tree.nodes[aid]["copies"])
            v = _compute_loss_values(all_v, self.evo_tree.nodes[aid]["copies"])
            h = _compute_loss_values(all_h, self.evo_tree.nodes[aid]["copies"])
            p = _compute_loss_values(all_p, self.evo_tree.nodes[aid]["copies"])
            score = fit / (np.log2(v + p))
            print(aid, version, "S:", score, "F:", fit, "V:", v, "P:", p, "H:", h)
            return score

        self.base_agent.sort(key=_val, reverse=True)
        killed = self.base_agent[num_survivors:]
        print("")
        for k in killed:
            if k.id in self.value_opitmizers:
                self.value_opitmizers.pop(k.id)
                self.policy_opitmizers.pop(k.id)
        self.base_agent = self.base_agent[:num_survivors]

    def integrate(self, new_agents, stats):
        # survivors = self.survival(new_agents)
        #num_survivors = len(survivors)
        alive = set(self.base_agent)
        all_agents = alive.union(set(new_agents))
        # updates evo tree with stats
        for a in all_agents:
            id = a.id
            if a not in alive:
                print(a.id, "went extinct after", a.version, "generations")
                continue
                # self.base_agent.append(a)
            elif id not in stats:
                continue
            if stats[id]["failure"]:
                print("FAILURE DETECTED: ", id)
                alive.remove(a)
                self.base_agent = list(alive)
            self.evo_tree.nodes[id]["fitness"].append(stats[id]["fitness"])
            self.evo_tree.nodes[id]["vloss"].append(stats[id]["value_loss"])
            self.evo_tree.nodes[id]["ploss"].append(stats[id]["policy_loss"])
            self.evo_tree.nodes[id]["entropy"].append(stats[id]["entropy"])
            self.evo_tree.nodes[id]["copies"].append(stats[id]["copies"])
        self.survival()
        # apply gradients
        survivor_fitness = []
        survivor_v_loss = []
        survivor_p_loss = []

        for i in range(len(self.base_agent)):
            id = self.base_agent[i].id
            if id not in stats:
                continue
            # apply gradients
            self.value_opitmizers[id].zero_grad()
            self.policy_opitmizers[id].zero_grad()
            grads = stats[id]["gradient"]
            for j, g in enumerate(grads):
                self.last_grad[j] = .4 * self.last_grad[j] + .6 * g
            self.base_agent[i].set_grad(self.last_grad)  # sets parameter gradient attributes
            before_plast = self.base_agent[i].core_model.edge.chan_map.detach().clone()
            self.value_opitmizers[id].step()
            self.policy_opitmizers[id].step()
            self.base_agent[i].version += 1
            after_plast = self.base_agent[i].core_model.edge.chan_map.detach().clone()
            print(id, self.base_agent[i].version, "change: ", torch.sum(torch.abs(after_plast - before_plast)))
            survivor_fitness.append(stats[id]["fitness"])
            survivor_v_loss.append(stats[id]["value_loss"])
            survivor_p_loss.append(stats[id]["policy_loss"])

        if len(survivor_fitness) <= 0:
            print("No Survivor History!")
        else:
            self.fitness_hist.append(np.max(survivor_fitness))
            self.value_loss_hist.append(np.min(survivor_v_loss))
            self.policy_loss_hist.append(np.min(survivor_p_loss))

        num_survivors = len(self.base_agent)
        # replace the deceased with random combinations of survivors.
        next_gen = []
        for i in range(self.num_base - num_survivors):
            parent1 = random.choice(self.base_agent)
            parent2 = random.choice(self.base_agent)
            child = self.multiclone(parent1, parent2, equal=True)
            if self.evo_tree.has_node(child.id):
                self.evo_tree.remove_node(child.id)
            # v_optim[child.id] = v_optim[parent1.id]
            # p_optim[child.id] = p_optim[parent1.id]
            for p in [parent1, parent2]:
                hist_size = len(self.evo_tree.nodes[parent1.id]["fitness"])
                # adjust weight by number of copies used
                cp = self.evo_tree.nodes[p.id]["copies"]
                fit = _compute_loss_values(self.evo_tree.nodes[p.id]["fitness"], cp) / 2
                v = _compute_loss_values(self.evo_tree.nodes[p.id]["vloss"], cp) / 2
                pl = _compute_loss_values(self.evo_tree.nodes[p.id]["ploss"], cp) / 2
                h = _compute_loss_values(self.evo_tree.nodes[p.id]["entropy"], cp) / 2
                fit = fit - .01
                v = v * 1.2
                if child.id in self.evo_tree.nodes:
                    self.evo_tree.nodes[child.id]["fitness"][-1] += fit
                    self.evo_tree.nodes[child.id]["vloss"][-1] += v
                    self.evo_tree.nodes[child.id]["ploss"][-1] += pl
                    self.evo_tree.nodes[child.id]["entropy"][-1] += h
                    self.evo_tree.nodes[child.id]["copies"][-1] += np.mean(cp) / 2
                else:
                    self.evo_tree.add_node(child.id, fitness=[fit], vloss=[v], ploss=[pl],
                                           copies=[np.mean(cp) / 2], entropy=[h])
                self.evo_tree.add_edge(p.id, child.id)
            self._add_optimizer_set(child)
            next_gen.append(child)
        self.base_agent.extend(next_gen)

    def _add_optimizer_set(self, a):
        aid = a.id
        if aid not in self.value_opitmizers:
            value_lr = float(np.power(10, random.random() * (self.log_max_lr - self.log_min_lr) + self.log_min_lr))
            policy_lr = float(np.power(10, random.random() * (self.log_max_lr - self.log_min_lr) + self.log_min_lr))
            self.value_opitmizers[a.id] = torch.optim.Adam(a.core_model.parameters() + [a.value_decoder, a.input_encoder],
                                                           lr=value_lr)
            self.policy_opitmizers[a.id] = torch.optim.Adam(a.core_model.parameters() + [a.policy_decoder, a.input_encoder],
                                                            lr=policy_lr)

    def spawn_visualization_worker(self, mp=True):
        # select current best base agent
        use_agent = [max(self.base_agent, key=lambda x: x.fitness)]
        copies = [2]
        if mp:
            p = Process(target=episode, args=(use_agent, copies, 200, 400, 20, True, "cpu"))
            return p
        else:
            episode(use_agent, copies, 200, 400, 20, True, "cpu")
            return

    def save_model(self, iter, fbase: str):
        if not os.path.isdir(fbase):
            os.mkdir(fbase)
        v = np.log2(_compute_loss_values(self.value_loss_hist))
        v = round(float(v), 2)
        package = {"agents": self.base_agent,
                   "v_optim": self.value_opitmizers,
                   "p_optim": self.policy_opitmizers,
                   "tree": self.evo_tree,
                   "fit_hist": self.fitness_hist,
                   "val_hist": self.value_loss_hist,
                   "p_hist": self.policy_loss_hist}
        fname = os.path.join(fbase, "snap_" + str(iter) + "_" + str(v) + "_.pkl")
        with open(fname, "wb") as file:
            pickle.dump(package, file)

    def load_model(self, fpath):
        # depackage
        with open(fpath, "rb") as f:
            p = pickle.load(f)
        self.evo_tree = p["tree"]
        agent_scores = [_compute_loss_values(self.evo_tree.nodes[a]["vloss"]) for a in p["agents"]]
        max_index = agent_scores.index(min(agent_scores))
        seed_agent = p["agents"][max_index]
        self.base_agent = seed_agent.clone(fuzzy=False)
        self.fitness_hist = p["fit_hist"]
        self.value_loss_hist = p["val_hist"]
        self.policy_loss_hist = p["p_hist"]

    def controller(self, mp=True, disp_iter=500, fbase="/Users/loggiasr/Projects/ReIntAI/models/evo_7"):
        num_workers = self.num_workers
        workers = {}
        epoch = 0
        fail = False
        while (epoch <= self.epochs and not fail) or len(workers) > 0:
            # time.sleep(.05)
            to_remove = []
            if mp:
                for k in workers.keys():
                    if not workers[k].is_alive():
                        print("Killing worker", k)
                        workers[k].close()
                        to_remove.append(k)
                for k in to_remove:
                    workers.pop(k)
                if len(workers) < num_workers and epoch <= self.epochs:
                    pid = "".join(random.choices("ABCDEFG1234567", k=5))
                    if (epoch) % disp_iter == 0:
                        print("Episode Display Worker", pid)
                        if epoch != 0:
                            self.save_model(epoch, fbase)
                        p = self.spawn_visualization_worker()
                    else:
                        print("Worker", pid, "handling epoch", epoch)
                        p = self.spawn_worker()
                    workers[pid] = p
                    p.start()
                    epoch += 1
            else:
                self.spawn_worker(mp=False)
                epoch += 1
            if not self.integration_q.empty():
                ret_agents, stats = self.integration_q.get(block=True)  # , v_optims, p_optims
                self.integrate(ret_agents, stats)
                #self.visualize()
                if (epoch + 1) % (disp_iter // 10) == 0:
                    self.visualize()
        print("DONE: one last visualization...")
        # save models
        for a in self.base_agent:
            name = a.id
            fitness = round(a.fitness, 2)
            with open("../models/" + name + "_" + str(a.core_model.num_nodes) + "_" + str(fitness) + ".pkl", "wb") as f:
                pickle.dump(a.detach(), f)
        self.visualize()
        self.spawn_visualization_worker(mp=True)
        plt.show(block=True)

    def visualize(self):
        self.axs[0].cla()
        self.axs[1].cla()
        self.axs[2].cla()
        self.axs[0].plot(np.log2(uniform_filter1d(np.array(self.value_loss_hist), size=self.num_workers)))
        self.axs[1].plot(np.log2(uniform_filter1d(np.array(self.policy_loss_hist)), size=self.num_workers))
        self.axs[2].plot(np.array(self.fitness_hist))
        # self.tree_axs.cla()
        # nx.draw(self.evo_tree, ax=self.tree_axs, with_labels=True)
        plt.pause(.05)


if __name__=="__main__":
    agent = WaterworldAgent(num_nodes=4, spatial=7, channels=3, input_channels=3)
    with open("/Users/loggiasr/Projects/ReIntAI/models/wworld_pretrain7.pkl", "rb") as f:
        in_enc = pickle.load(f)
    agent.input_encoder = in_enc
    base = pickle
    evo = EvoController(seed_agent=agent, epochs=10000, num_base=3, num_workers=6,
                        min_agents=3, max_agents=5, min_gen=1, max_gen=1, log_min_lr=-8, log_max_lr=-2)
    evo.controller(mp=False, fbase="/Users/loggiasr/Projects/ReIntAI/models/evo_7")

