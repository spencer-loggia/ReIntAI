
import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import torch
import torch.nn.functional as F


def return_from_reward(rewards, gamma):
    # Calculate the returns using cumulative sum and then reverse the order
    returns = torch.flip(torch.cumsum(torch.flip(rewards, [0]) * gamma, dim=0), [0])
    return returns


class ActorCritic:

    def __init__(self, gamma, alpha, stat_gamma=.98):
        self.gamma = gamma
        self.alpha = alpha
        self.mean = 0.
        self.std = 1.
        self._stat_gamma = stat_gamma
        self.count = 0.

    def loss(self, rewards, value_estimates, log_probs, entropies):
        returns = return_from_reward(rewards, self.gamma)[:-15]
        sg = self._stat_gamma
        self.count += 1
        sg = sg * (1 - 1 / self.count)
        self.mean = self.mean * sg + (1 - sg) * returns.mean().detach().item()
        self.std = self.std * sg + (1 - sg) * (returns.std().detach().item())
        returns = (returns - self.mean) / (self.std + 1e-8)
        # Calculate the critic loss
        critic_loss = F.mse_loss(value_estimates[:-15].squeeze(), returns)
        # Compute the advantage
        advantages = returns - value_estimates[:-15].detach().squeeze()
        # Calculate the actor loss incorporating the entropy term
        actor_loss = -(log_probs[:-15] * advantages + self.alpha * entropies[:-15]).mean()
        return critic_loss, actor_loss

    def __add__(self, other):
        sg = self._stat_gamma
        self.count += 1
        sg = sg * (1 - 1 / self.count)
        self.mean = (self.mean * sg + (1 - sg) * other.mean)
        self.std = (sg * self.std + (1 - sg) * other.std)
        self.count = max(self.count, other.count)
        return self


class Reinforce:

    def __init__(self, gamma, alpha, stat_gamma=.98):
        self.gamma = gamma
        self.alpha = alpha
        self.mean = 0.
        self.std = 1.
        self._stat_gamma = stat_gamma
        self.count = 0.

    def loss(self, rewards, log_probs, entropies):
        returns = return_from_reward(rewards, self.gamma)
        sg = self._stat_gamma
        self.count += 1
        sg = sg * (1 - 1 / self.count)
        self.mean = self.mean * sg + (1 - sg) * returns.mean().detach().item()
        self.std = self.std * sg + (1 - sg) * (returns.std().detach().item())
        returns = (returns - self.mean) / (self.std + 1e-8)
        # Calculate the actor loss incorporating the entropy term
        policy_loss = -(log_probs * returns + self.alpha * entropies).mean()
        return policy_loss

    def __add__(self, other):
        sg = self._stat_gamma
        self.count += 1
        sg = sg * (1 - 1 / self.count)
        self.mean = (self.mean * sg + (1 - sg) * other.mean)
        self.std = (sg * self.std + (1 - sg) * other.std)
        self.count = max(self.count, other.count)
        return self
