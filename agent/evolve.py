import math
import os.path
import pickle
import random
import networkx

import matplotlib
import matplotlib.pyplot as plt
import randomname

matplotlib.use('Qt5Agg')

import numpy as np
import torch
from torch.multiprocessing import Queue, Process
from collections import deque

from agent.agents import WaterworldAgent, DisjointWaterWorldAgent
from agent.reward_functions import Reinforce, ActorCritic
from agent.exist import local_evolve, episode
from scipy.ndimage import uniform_filter1d


def _compute_loss_values(arr, copies=None, window=15):
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

def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return


class _pseudo_queue(deque):
    def __init__(self):
        super().__init__()
    def get(self, *args, **kwargs):
        return self.popleft()

    def put(self, item, *args, **kwargs):
        self.append(item)

    def empty(self, *args, **kwargs):
        return len(self) == 0

    def close(self, *args, **kwargs):
        pass


class EvoController:
    def __init__(self, seed_agent: WaterworldAgent, epochs=10, num_base=4,
                 min_gen=10, max_gen=30, min_agents=3, max_agents=8,
                 log_min_lr=-13., log_max_lr=-8., num_workers=6, worker_device="cpu", viz=True,
                 algo="a3c", start_epsilon=1.0, inverse_eps_decay=4000):
        self.num_base = num_base
        self.log_min_lr = log_min_lr
        self.log_max_lr = log_max_lr
        self.epochs = epochs
        self.min_gen = min_gen
        self.max_gen = max_gen
        self.min_agents = min_agents
        self.max_agents = max_agents
        self.last_grad = [0. for _ in seed_agent.parameters()]
        self.viz=viz
        self.algo = algo
        if algo == "a3c":
            self.reward_function = ActorCritic(gamma=.98, alpha=.25)
        elif algo == "reinforce":
            self.reward_function = Reinforce(gamma=.98, alpha=.25)
        else:
            raise ValueError
        self.full_count = 0
        self.epsilon = start_epsilon
        self.decay = -1 / inverse_eps_decay
        if type(seed_agent) is WaterworldAgent:
            self.disjoint_critic = False
        elif type(seed_agent) is DisjointWaterWorldAgent:
            self.disjoint_critic = True
        else:
            raise TypeError

        self.num_workers = num_workers
        self.sensors = seed_agent.num_sensors
        self.base_agent = [seed_agent.clone(fuzzy=False)]
        self.value_opitmizers = {}
        self.policy_opitmizers = {}
        self.worker_device = worker_device
        self.num_integrations = 0
        self.evo_tree = networkx.DiGraph()
        self.evo_tree.add_node(seed_agent.id, fitness=[], vloss=[], ploss=[], copies=[], entropy=[])
        self.value_loss_hist = []
        self.policy_loss_hist = []
        self.fitness_hist = []

        if self.viz:
            # local display figure
            self.fig, self.axs = plt.subplots(3)
            self.fig.suptitle("Loss Curves " + self.algo + " disjoint critic " + str(self.disjoint_critic))
            self.axs[0].set_ylabel("Value loss")
            self.axs[1].set_ylabel("Policy loss")

            # global loss display figures
            #self.global_fig, self.global_axs = plt.subplots(3)
            self.axs[2].set_ylabel("Fitness")
            self.axs[2].set_ylim((-.05, .2))
            plt.show()

        # self.global_fig.suptitle("Overall Loss Progress")

        # tree display figure
        # self.tree_fig, self.tree_axs = plt.subplots(1)
        # self.tree_fig.suptitle("Evolutionary Tree")

    def spawn_worker(self, integration_q, pid, mp=True):
        for i in range(len(self.base_agent)):
            self._add_optimizer_set(self.base_agent[i])
        num_gens = random.randint(self.min_gen, self.max_gen)
        num_agents = random.randint(self.min_agents, self.max_agents)
        select_base_idx = np.random.choice(np.arange(len(self.base_agent)), size=num_agents)
        use_base_idx, copies = np.unique(select_base_idx, return_counts=True)
        use_base = []
        local_eps = max(self.decay * self.full_count + self.epsilon, .05)
        for i in use_base_idx:
            a = self.base_agent[i].clone(fuzzy=False)
            if random.random() < .1:
                a.epsilon = 1.0
            else:
                a.epsilon = local_eps
            use_base.append(a)
        if self.algo == "a3c" and (self.full_count < 250):
            train_critic = True
            train_actor = False
        else:
            train_critic = True
            train_actor = True


        print("OPTIM:", num_gens, "generations,", num_agents, "agents of types:", [a.id for a in use_base])
        train_critic_random_only = not self.disjoint_critic
        if mp:
            p = Process(target=local_evolve,
                        args=(integration_q, num_gens, use_base, copies.tolist(), self.reward_function, train_actor,
                              train_critic, train_critic_random_only, pid))
            return p
        else:
            local_evolve(integration_q, num_gens, use_base, copies.tolist(), self.reward_function, train_actor,
                         train_critic, train_critic_random_only, pid)
            return None

    def multiclone(self, agent1: WaterworldAgent, agent2: WaterworldAgent, equal=False):
        new_agent = type(agent1)(num_nodes=agent1.core_model.num_nodes,
                                    channels=agent1.channels, spatial=agent1.spatial,
                                    kernel=agent1.core_model.edge.kernel_size, sensors=agent1.num_sensors,
                                    action_dim=agent1.action_dim,
                                    device=agent1.device, input_channels=agent1.input_channels)
        with torch.no_grad():
            if equal:
                new_core_1 = agent1.core_model.clone(fuzzy=False)
                lincomb = .5
            else:
                new_core_1 = agent1.core_model.clone(fuzzy=True)
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
        kill_prob = 1 / (4 * self.num_workers)
        if random.random() > kill_prob:
            return
        # num_survivors = min(self.num_base - 1, math.ceil(self.num_base * .75))
        num_survivors = max(1, math.floor(self.num_base * .75))

        print("Agent pool")

        def _val(a: WaterworldAgent):
            aid = a.id
            version = a.version
            all_ = self.evo_tree.nodes[aid]["fitness"]
            all_v = self.evo_tree.nodes[aid]["vloss"]
            all_p = self.evo_tree.nodes[aid]["ploss"]
            fit = _compute_loss_values(all_, self.evo_tree.nodes[aid]["copies"])
            v = _compute_loss_values(all_v, self.evo_tree.nodes[aid]["copies"])
            p = _compute_loss_values(all_p, self.evo_tree.nodes[aid]["copies"])
            score = fit
            print(aid, version, "S:", score, "F:", fit, "V:", v, "P:", p)
            return score

        self.base_agent.sort(key=_val, reverse=True)
        killed = self.base_agent[num_survivors:]
        print("")
        for k in killed:
            if k.id in self.value_opitmizers:
                self.value_opitmizers.pop(k.id)
                self.policy_opitmizers.pop(k.id)
        self.base_agent = self.base_agent[:num_survivors]

    def integrate(self, stats):
        # survivors = self.survival(new_agents)
        alive = set(self.base_agent)
        # updates evo tree with stats
        for a in self.base_agent:
            id = a.id
            if id not in stats:
                continue
            if stats[id]["failure"]:
                print("FAILURE DETECTED: ", id)
                alive.remove(a)
                self.base_agent = list(alive)
            self.evo_tree.nodes[id]["fitness"].append(stats[id]["fitness"])
            self.evo_tree.nodes[id]["vloss"].append(stats[id]["value_loss"])
            self.evo_tree.nodes[id]["ploss"].append(stats[id]["policy_loss"])
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
                self.last_grad[j] = .7 * self.last_grad[j] + .3 * g
            self.base_agent[i].set_grad(self.last_grad)  # sets parameter gradient attributes
            before_plast = self.base_agent[i].core_model.edge.chan_map.detach().clone()
            self.value_opitmizers[id].step()
            self.policy_opitmizers[id].step()
            self.base_agent[i].version += 1
            after_plast = self.base_agent[i].core_model.edge.chan_map.detach().clone()
            change = torch.sum(torch.abs(after_plast - before_plast))
            print(id, self.base_agent[i].version, "change: ", change)
            survivor_fitness.append(stats[id]["fitness"])
            survivor_v_loss.append(stats[id]["value_loss"])
            survivor_p_loss.append(stats[id]["policy_loss"])

        if len(survivor_fitness) <= 0:
            print("No Survivor History!")
        else:
            self.fitness_hist.append(np.mean(survivor_fitness))
            self.value_loss_hist.append(np.mean(survivor_v_loss))
            self.policy_loss_hist.append(np.mean(survivor_p_loss))

        num_survivors = len(self.base_agent)
        # replace the deceased with random combinations of survivors.
        next_gen = []
        for i in range(self.num_base - num_survivors):
            parent1 = random.choice(self.base_agent)
            parent2 = random.choice(self.base_agent)
            if random.random() < 1.0:
                child = self.multiclone(parent1, parent2, equal=True)
            else:
                child = self.multiclone(parent1, parent2)
            child.epsilon = random.random() * .1
            if self.evo_tree.has_node(child.id):
                self.evo_tree.remove_node(child.id)
            for p in [parent1, parent2]:
                hist_size = len(self.evo_tree.nodes[parent1.id]["fitness"])
                # adjust weight by number of copies used
                cp = self.evo_tree.nodes[p.id]["copies"]
                fit = _compute_loss_values(self.evo_tree.nodes[p.id]["fitness"], cp) / 2
                v = _compute_loss_values(self.evo_tree.nodes[p.id]["vloss"], cp) / 2
                pl = _compute_loss_values(self.evo_tree.nodes[p.id]["ploss"], cp) / 2
                fit = fit - .01
                v = v * 1.0
                if child.id in self.evo_tree.nodes:
                    self.evo_tree.nodes[child.id]["fitness"][-1] += fit
                    self.evo_tree.nodes[child.id]["vloss"][-1] += v
                    self.evo_tree.nodes[child.id]["ploss"][-1] += pl
                    self.evo_tree.nodes[child.id]["copies"][-1] += np.mean(cp) / 2
                else:
                    self.evo_tree.add_node(child.id, fitness=[fit], vloss=[v], ploss=[pl],
                                           copies=[np.mean(cp) / 2])
                self.evo_tree.add_edge(p.id, child.id)
            self._add_optimizer_set(child)
            next_gen.append(child)
        self.base_agent.extend(next_gen)

    def _add_optimizer_set(self, a):
        aid = a.id
        if aid not in self.value_opitmizers:
            lr = float(np.power(10, random.random() * (self.log_max_lr - self.log_min_lr) + self.log_min_lr))
            value_lr = lr
            policy_lr = lr
            if self.disjoint_critic:
                self.value_opitmizers[a.id] = torch.optim.Adam([a.value_decoder, a.input_encoder], lr=.8 * value_lr)
            else:
                self.value_opitmizers[a.id] = torch.optim.Adam(a.core_model.parameters() + [a.value_decoder, a.input_encoder],
                                                               lr=value_lr)
            self.policy_opitmizers[a.id] = torch.optim.Adam(a.core_model.parameters() + [a.policy_decoder, a.input_encoder],
                                                            lr=policy_lr)

    def spawn_visualization_worker(self, mp=True):
        # select current best base agent
        use_agent = [max(self.base_agent, key=lambda x: x.fitness)]
        decay_by = 4000
        use_agent[0].epsilon = max(-(1/decay_by) * self.full_count + 1.0, .01)
        copies = [1]
        if mp:
            p = Process(target=episode, args=(use_agent, copies, 400, 400, 20, True, "cpu"))
            return p
        else:
            episode(use_agent, copies, 400, 400, 20, True, "cpu")
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
                   "p_hist": self.policy_loss_hist,
                   "r_fxn": self.reward_function,
                   "count": self.full_count}
        fname = os.path.join(fbase, "snap_" + str(iter) + "_" + str(v) + "_.pkl")
        with open(fname, "wb") as file:
            pickle.dump(package, file)

    def load_model(self, fpath):
        # depackage
        with open(fpath, "rb") as f:
            p = pickle.load(f)
        self.evo_tree = p["tree"]
        self.base_agent = p["agents"]
        self.fitness_hist = p["fit_hist"]
        self.value_loss_hist = p["val_hist"]
        self.policy_loss_hist = p["p_hist"]
        try:
            rf = p["r_fxn"]
            rf.alpha = .01
            self.full_count = p["count"]
            # don't directly assign so we can change rfs
            self.reward_function.count = rf.count
            self.reward_function.mean = rf.mean
            self.reward_function.std = rf.std
        except KeyError:
            print("No reward fxn in saved dict.")

    def controller(self, mp=True, disp_iter=500, fbase="/Users/loggiasr/Projects/ReIntAI/models/evo_7"):
        num_workers = self.num_workers
        workers = {}
        epoch = 0
        fail = False
        to_kill = set()
        if mp:
            integration_q = Queue(maxsize=100)
        else:
            integration_q = _pseudo_queue()
        while (epoch <= self.epochs and not fail) or len(workers) > 0:
            # time.sleep(.05)
            # to_remove = []
            if mp:
                for k in to_kill:
                    workers[k].join()
                    workers.pop(k)
                to_kill = set()
                if len(workers) < num_workers and epoch <= self.epochs:
                    pid = "".join(random.choices("ABCDEFG1234567", k=5))
                    if (epoch) % disp_iter == 0:
                        print("Episode Display Worker", pid)
                        if epoch != 0:
                            self.save_model(epoch, fbase)
                        p = self.spawn_visualization_worker(mp=False)
                    else:
                        print("Worker", pid, "handling epoch", epoch)
                        p = self.spawn_worker(integration_q, pid)
                        workers[pid] = p
                        p.start()
                    epoch += 1
                    self.full_count += 1
            else:
                if self.viz and (epoch + 1) % disp_iter == 0:
                    self.spawn_visualization_worker(mp=False)
                else:
                    self.spawn_worker(integration_q, 0, mp=False)
                epoch += 1
                self.full_count += 1
            if not integration_q.empty():
                stats, rf, pid = integration_q.get(block=True)  # , v_optims, p_optims
                to_kill.add(pid)
                if stats is None:
                    print("Worker", pid, "FAILED")
                    continue
                self.reward_function = self.reward_function + rf
                self.integrate(stats)
                if self.viz and (epoch + 1) % (disp_iter // 10) == 0:
                    self.visualize()
        for k in workers.keys():
            workers[k].join()
        print("DONE: one last visualization...")

        if self.viz:
            self.visualize()
            self.spawn_visualization_worker(mp=False)
            plt.show(block=True)
        integration_q.close()

    def visualize(self):
        self.axs[0].cla()
        self.axs[1].cla()
        self.axs[2].cla()
        self.axs[0].plot(np.log2(uniform_filter1d(np.array(self.value_loss_hist), size=5 * self.num_workers)))
        self.axs[1].plot(uniform_filter1d(np.array(self.policy_loss_hist), size=5 * self.num_workers))
        self.axs[2].plot(uniform_filter1d(np.array(self.fitness_hist), size=5 * self.num_workers))
        mypause(.05)


