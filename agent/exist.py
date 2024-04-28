import random

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')

import numpy as np
import torch

from pettingzoo.sisl import waterworld_v4

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
        obs_fig, obs_ax = plt.subplots(1)
        env = waterworld_v4.parallel_env(render_mode="human", n_pursuers=num_agents, n_coop=1,
                                         n_sensors=sensors, max_cycles=cycles, speed_features=False,
                                         pursuer_max_accel=.25, encounter_reward=0.1, food_reward=8.0,
                                         poison_reward=-3.5, thrust_penalty=-.001)
    else:
        env = waterworld_v4.parallel_env(n_pursuers=num_agents, n_coop=1, n_sensors=sensors,
                                         max_cycles=cycles, speed_features=False, pursuer_max_accel=.25,
                                         encounter_reward=0.1, food_reward=8.0, poison_reward=-3.5,
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
            inst_r = 0
            if len(agent_dict[agent]["inst_r"]) > 0:
                inst_r = agent_dict[agent]["inst_r"][-1]
            c1, c2, v_hat = agent_dict[agent]["model"](torch.from_numpy(observations[agent]) + .00001, inst_r + .00001)
            if human:
                m = agent_dict[agent]["model"]
                with torch.no_grad():
                    obs_ax.cla()
                    proj_obs = (torch.from_numpy(observations[agent])) @ m.input_encoder
                    proj_obs = torch.sigmoid(proj_obs)
                    proj_obs = (proj_obs - torch.min(proj_obs))
                    proj_obs = proj_obs / torch.max(proj_obs)
                    proj_obs = proj_obs.view(m.input_channels, m.spatial, m.spatial)
                    obs_ax.imshow(proj_obs.permute((1, 2, 0)).detach().cpu().numpy())
                    mypause(.05)
            # c1, c2, v_hat = agent_dict[agent]["model"](torch.from_numpy(observations[agent]) + .0001)
            with torch.no_grad():
                if torch.isnan(c1 + c2 + v_hat).any():
                    agent_dict[agent]["failure"] = True
                    agent_dict[agent]["terminate"] = True
            if not agent_dict[agent]["terminate"]:
                x_dist = torch.distributions.Beta(concentration0=c1[0], concentration1=c1[1])
                y_dist = torch.distributions.Beta(concentration0=c2[0], concentration1=c2[1])
                if hasattr(agent_dict[agent]["model"], "epsilon") and random.random() < agent_dict[agent]["model"].epsilon:
                    actions[agent] = torch.rand((2,), device=device)
                else:
                    # rdist = torch.distributions.Beta(alpha, beta)
                    action_x = x_dist.sample()
                    action_y = y_dist.sample()
                    action = torch.stack([action_x, action_y]) - .5
                # action = torch.stack([action_mag * torch.cos(action_rad), action_mag * torch.sin(action_rad)])
                # print("ACTION", action)
                likelihood_x = x_dist.log_prob(action_x)
                likelihood_y = y_dist.log_prob(action_y)
                entropy = x_dist.entropy() + y_dist.entropy()
                # if entropy > 2000:
                #     from matplotlib import pyplot as plt
                #     with torch.no_grad():
                #         x = 2 * torch.pi * torch.arange(1000) / 1000
                #         l = torch.stack([adist.log_prob(xi) for xi in x])
                #     plt.ylim((-100, 0.1))
                #     plt.plot(x.numpy(), l.numpy())
                #     plt.show()
                agent_dict[agent]["action_likelihood"].append(likelihood_x + likelihood_y)
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


def local_evolve(q, generations, base_agents, copies, reward_function, train_act=True, train_critic=True, device="cpu"):
    num_base = len(base_agents)
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

    for gen in range(generations):
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
                val_loss, policy_loss = reward_function.loss(torch.Tensor([0.] + agent_info["inst_r"], device=device),
                                                                       torch.concatenate(agent_info["value"], dim=0),
                                                                       torch.stack(agent_info["action_likelihood"], dim=0),
                                                                       torch.stack(agent_info["entropy"], dim=0))
                # policy_loss = reward_function.loss(torch.Tensor([0.] + agent_info["inst_r"], device=device),
                #                                                        torch.stack(agent_info["action_likelihood"], dim=0),
                #                                                        torch.stack(agent_info["entropy"], dim=0))

               # val_loss = torch.Tensor([8.])
                if torch.isnan(val_loss + policy_loss):
                    print("NaN is gradient!", agent_info["base_name"])
                stat_tracker[agent_info["base_name"]]["value_loss"][-1].append(val_loss.detach().cpu().item())
                stat_tracker[agent_info["base_name"]]["policy_loss"][-1].append(policy_loss.detach().cpu().item())
                stat_tracker[agent_info["base_name"]]["copies"] += 1
                a = .05
                b = .05
                if train_act:
                    b = .005
                if train_critic:
                    a = .005
                total_loss[agent_info["base_index"]] = total_loss[agent_info["base_index"]] + a * val_loss + b * policy_loss
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
                reg = torch.sum(torch.abs(a.input_encoder))
                reg = reg + torch.sum(torch.abs(a.core_model.edge.init_weight))
                reg = reg + torch.sum(torch.abs(a.core_model.edge.chan_map))
                reg = reg + torch.sum(torch.abs(a.value_decoder))
                reg = reg + torch.sum(torch.abs(a.policy_decoder))
                total_loss[i] = total_loss[i] + .0001 * reg
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
    # average and cast to numpy
    for k in stat_tracker.keys():
        if not stat_tracker[k]["failure"]:
            for pid in range(len(stat_tracker[k]["gradient"])):
                stat_tracker[k]["gradient"][pid] *= 1e-1
            stat_tracker[k]["value_loss"] = np.nanmean(np.array(stat_tracker[k]["value_loss"], dtype=float))
            stat_tracker[k]["policy_loss"] = np.nanmean(np.array(stat_tracker[k]["policy_loss"], dtype=float))
            if stat_tracker[k]["fitness"] is not None:
                stat_tracker[k]["fitness"] = np.mean(stat_tracker[k]["fitness"])

    q.put(([a.detach() for a in base_agents], stat_tracker, reward_function))
    return

