import random
import time

import numpy as np
import torch

from pettingzoo.sisl import waterworld_v4

def episode(base_agents, copies, min_cycles=800, max_cycles=800, sensors=20, human=False, device="cpu", max_acc=.5,
            action_dist="weighted_dist"):
    """
    Function to run launch and take action in the waterworld environment
    :param base_agents: Agent species that are present in this environment (e.g. unique parameter set)
    :param copies: Number of copies of each base type
    :param min_cycles: minimum number of max cylces to run
    :param max_cycles: maximum number of max cycles to run
    :param sensors: number of environment sensors on agent.
    :param human: whether to display environment runtime on screen (slow, locks process)
    :param device: device ot use for gradient computation
    :param max_acc: maximum agent acceleration in environment
    :return:
    """
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
                                         pursuer_max_accel=max_acc, encounter_reward=0.1, food_reward=6.0,
                                         poison_reward=-3.5, thrust_penalty=-.001)
    else:
        env = waterworld_v4.parallel_env(n_pursuers=num_agents, n_coop=1, n_sensors=sensors,
                                         max_cycles=cycles, speed_features=False, pursuer_max_accel=max_acc,
                                         encounter_reward=0.1, food_reward=6.0, poison_reward=-3.5,
                                         thrust_penalty=-.001)
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
                                      "is_random": [],
                                      "entropy": [],
                                      "inst_r": [],
                                      "counts": 0,
                                      "value": [],
                                      "terminate": False,
                                      "failure": False}
            env_agent_index += 1

    observations, infos = env.reset()

    while env.agents:
        # this is where you would insert your policy
        actions = {}
        for agent in env.agents:
            inst_r = 0
            if len(agent_dict[agent]["inst_r"]) > 0:
                inst_r = agent_dict[agent]["inst_r"][-1]
            c1, c2, v_hat = agent_dict[agent]["model"].forward(torch.from_numpy(observations[agent]) + .00001, inst_r + .00001)
            # c1, c2, v_hat = agent_dict[agent]["model"](torch.from_numpy(observations[agent]) + .0001)
            with torch.no_grad():
                if torch.isnan(c1 + c2 + v_hat).any():
                    agent_dict[agent]["failure"] = True
            x_dist = torch.distributions.Beta(concentration0=c1[0], concentration1=c1[1])
            y_dist = torch.distributions.Beta(concentration0=c2[0], concentration1=c2[1])
            if random.random() < agent_dict[agent]["model"].epsilon:
                agent_dict[agent]["is_random"].append(True)
                action = torch.rand((2,), device=device)
                action_x = action[0]
                action_y = action[1]
                action = action * 2 * max_acc - max_acc
            else:
                agent_dict[agent]["is_random"].append(False)
                action_x = x_dist.sample()
                action_y = y_dist.sample()
                action = torch.stack([action_x, action_y]) * 2 * max_acc - max_acc
            likelihood_x = x_dist.log_prob(action_x)
            likelihood_y = y_dist.log_prob(action_y)
            entropy = x_dist.entropy() + y_dist.entropy()

            agent_dict[agent]["action_likelihood"].append(likelihood_x + likelihood_y)
            agent_dict[agent]["entropy"].append(entropy)
            agent_dict[agent]["value"].append(v_hat.clone())
            actions[agent] = action.detach().cpu().numpy()
        try:
            observations, rewards, terminations, truncations, infos = env.step(actions)
        except ValueError:
            for agent in env.agents:
                agent_dict[agent]["failure"] = True
                base = agent_dict[agent]["base_index"]
                scores[base] = None
        for i, agent in enumerate(agent_dict.keys()):
            base = agent_dict[agent]["base_index"]
            if terminations[agent]:
                scores[base] = None
                agent_dict[agent]["terminate"] = True
            if not agent_dict[agent]["terminate"]:
                reward = rewards[agent]
                agent_dict[agent]["inst_r"].append(reward)
                agent_dict[agent]["counts"] += 1
                scores[base] += float(reward)
    for agent in agent_dict.keys():
        if scores[agent_dict[agent]["base_index"]] is not None:
            scores[agent_dict[agent]["base_index"]] /= agent_dict[agent]["counts"]
    env.close()
    return agent_dict, scores


def local_evolve(q, pipe, generations, base_agents, copies, reward_function, train_act=True, train_critic=True, critic_random_only=False, proc=0, device="cpu"):
    try:
        num_base = len(base_agents)
        device = base_agents[0].device
        fail_tracker = [False for _ in range(num_base)]
        stat_tracker = {a.id: {"gradient": [0. for _ in a.parameters()],
                               "value_loss": [],
                               "policy_loss": [],
                               "entropy": [],
                               "fitness": [],
                               "copies": copies[i] * generations,
                               "failure": False} for i, a in enumerate(base_agents)}
        for gen in range(generations):
            h_int = False
            total_loss = [torch.tensor([0.], device=device) for _ in range(num_base)]
            gen_info, base_scores = episode(base_agents, copies, h_int, device=device)
            for a in stat_tracker.keys():
                stat_tracker[a]["value_loss"].append([])
                stat_tracker[a]["policy_loss"].append([])
                stat_tracker[a]["entropy"].append([])

            for agent in gen_info.keys():
                agent_info = gen_info[agent]
                if agent_info["terminate"]:
                    stat_tracker["skip"] = True
                    continue
                if not (agent_info["failure"]) and not stat_tracker[agent_info["base_name"]]["failure"]:
                    # whether to train critic only when random action is taken.
                    is_random = torch.tensor(agent_info["is_random"], device=device)

                    if reward_function.__name__ == "ActorCritic":
                        val_loss, policy_loss = reward_function.loss(torch.tensor(agent_info["inst_r"], device=device),
                                                                               torch.concat(agent_info["value"], dim=0),
                                                                               torch.stack(agent_info["action_likelihood"], dim=0),
                                                                               torch.stack(agent_info["entropy"], dim=0))
                    elif reward_function.__name__ == "Reinforce":
                        policy_loss = reward_function.loss(torch.tensor(agent_info["inst_r"], device=device),
                                                                               torch.stack(agent_info["action_likelihood"], dim=0),
                                                                               torch.stack(agent_info["entropy"], dim=0))
                        val_loss = torch.tensor([0.])
                    else:
                        raise ValueError("undefined reward function")

                   # val_loss = torch.Tensor([8.])
                    weight = max(len(agent_info["inst_r"]) - 15, 0) / 400
                    if torch.isnan(val_loss + policy_loss):
                        print("NaN is gradient!", agent_info["base_name"])
                        stat_tracker[agent_info["base_name"]]["failure"] = True
                        fail_tracker[agent_info["base_index"]] = True
                    stat_tracker[agent_info["base_name"]]["value_loss"][-1].append((val_loss / len(is_random)).detach().cpu().item())
                    stat_tracker[agent_info["base_name"]]["policy_loss"][-1].append((policy_loss / len(is_random)).detach().cpu().item())
                    stat_tracker[agent_info["base_name"]]["copies"] += weight
                    a = .01
                    b = .1
                    if not train_act:
                        b = 0.0
                    if not train_critic:
                        a = 0.0
                    total_loss[agent_info["base_index"]] = (total_loss[agent_info["base_index"]] + a * val_loss
                                                            + b * policy_loss)
                else:
                    agent_info["failure"] = True
                    stat_tracker[agent_info["base_name"]]["Failure"] = True
                    fail_tracker[agent_info["base_index"]] = True
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
            for i in range(num_base):
                if not fail_tracker[i]:
                    a = base_agents[i]
                    # reg = torch.sum(torch.abs(a.input_encoder))
                    # reg = reg + torch.sum(torch.abs(a.core_model.edge.init_weight))
                    # reg = reg + torch.sum(torch.abs(a.core_model.edge.chan_map))
                    # if train_critic:
                    #     reg = reg + torch.sum(torch.abs(a.value_decoder))
                    # if train_act:
                    #     reg = reg + torch.sum(torch.abs(a.policy_decoder))
                    # total_loss[i] = total_loss[i] + .0001 * reg
                    total_loss[i].backward()
                    for j, p in enumerate(a.parameters()):
                        if not torch.isnan(p.grad).any():
                            # send gradient to cpu
                            stat_tracker[a.id]["gradient"][j] += p.grad.detach().cpu()
                        else:
                            print("NaN grad", a.id)
                            stat_tracker[a.id]["failure"] = True
                            stat_tracker[a.id]["gradient"][j] += torch.zeros_like(p.grad)
                            fail_tracker[i] = True
        # average and cast to numpy
        for k in stat_tracker.keys():
            if not stat_tracker[k]["failure"]:
                for pid in range(len(stat_tracker[k]["gradient"])):
                    stat_tracker[k]["gradient"][pid] *= 1e-1
                stat_tracker[k]["value_loss"] = np.nanmean(np.array(stat_tracker[k]["value_loss"], dtype=float))
                stat_tracker[k]["policy_loss"] = np.nanmean(np.array(stat_tracker[k]["policy_loss"], dtype=float))
                if stat_tracker[k]["fitness"] is not None:
                    stat_tracker[k]["fitness"] = np.mean(stat_tracker[k]["fitness"])
        q.put((stat_tracker, reward_function, proc))
    except Exception as e:
        # on any exception we return the pid so proc can be killed
        print("CAUGHT in local_evolve\n", e, "\n")
        q.put((None, None, proc))
    if pipe is None:
        return
    elif pipe.recv():
        # wait for parent to signal done with data.
        return
    raise RuntimeError("Worker", proc, "was never signalled to die.")

