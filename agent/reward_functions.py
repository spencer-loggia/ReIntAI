import torch


def return_from_reward(rewards, gamma):
    """
    Compute the discounted returns for each timestep from a tensor of rewards.

    Parameters:
    - rewards (torch.Tensor): Tensor containing the instantaneous rewards.
    - gamma (float): Discount factor (0 < gamma <= 1).

    Returns:
    - torch.Tensor: Tensor containing the discounted returns.
    """
    # Initialize an empty tensor to store the returns
    returns = torch.zeros_like(rewards)

    # Variable to store the accumulated return, initialized to 0
    G = 0

    # Iterate through the rewards in reverse (from future to past)
    for t in reversed(range(len(rewards))):
        # Update the return: G_t = r_t + gamma * G_{t+1}
        G = rewards[t] + gamma * G
        returns[t] = G

    return returns



class ActorCritic:

    def __init__(self, gamma, alpha, stat_gamma=.98):
        self.gamma = gamma
        self.alpha = alpha
        self.mean = 0.
        self.std = 1.
        self._stat_gamma = stat_gamma
        self.count = 0.
        self.debug = False
        self.__name__ = "ActorCritic"

    def loss(self, rewards, value_estimates, log_probs, entropies, is_random=None):
        cutoff = max(16, len(rewards) - 15)
        if self.debug:
            entropies.register_hook(lambda grad: print("Grad H ", torch.abs(grad).sum()))
            log_probs.register_hook(lambda grad: print("Grad LogProb ", torch.abs(grad).sum()))
        returns = return_from_reward(rewards, self.gamma)[:cutoff]
        sg = self._stat_gamma
        self.count += 1
        sg = sg * (1 - 1 / self.count)
        mean = returns.mean().detach()
        std = returns.std().detach()
        if not torch.isnan(mean + std):
            self.mean = self.mean * sg + (1 - sg) * mean.item()
            self.std = self.std * sg + (1 - sg) * (std.item())
        # returns = (returns - self.mean) / (self.std + 1e-8)
        # compute advantages
        advantages = returns - value_estimates[:cutoff]
        # Calculate the critic loss, only where actions are random
        if is_random is not None:
            masked_td = advantages * is_random[:cutoff].float()
        else:
            masked_td = advantages
        critic_loss = torch.pow(masked_td, 2).sum()
        # critic_loss.register_hook(lambda grad: print("Grad C Loss", torch.abs(grad).sum()))
        # Calculate the actor loss incorporating the entropy term
        actor_loss = -(log_probs[:cutoff] * advantages.detach() + self.alpha * entropies[:cutoff]).sum()
        if torch.isnan(critic_loss + actor_loss):
            print("NAN, returns", returns)
            print("NAN, V", value_estimates)
            print("NAN, Prob", log_probs)
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
        self. __name__ = "Reinforce"

    def loss(self, rewards, log_probs, entropies):
        cutoff = max(16, len(rewards) - 15)
        returns = return_from_reward(rewards, self.gamma)[:cutoff]
        sg = self._stat_gamma
        self.count += 1
        sg = sg * (1 - 1 / self.count)
        mean = returns.mean().detach()
        std = returns.std().detach()
        if not torch.isnan(mean + std):
            self.mean = self.mean * sg + (1 - sg) * mean.item()
            self.std = self.std * sg + (1 - sg) * (std.item())
        returns = (returns - self.mean) / (self.std + 1e-8)
        # Calculate the actor loss incorporating the entropy term
        policy_loss = -(log_probs[:cutoff] * returns + self.alpha * entropies[:cutoff]).mean()
        if torch.isnan(policy_loss):
            print("NAN, returns", returns)
            print("NAN, Prob", log_probs)
        return policy_loss

    def __add__(self, other):
        sg = self._stat_gamma
        self.count += 1
        sg = sg * (1 - 1 / self.count)
        self.mean = (self.mean * sg + (1 - sg) * other.mean)
        self.std = (sg * self.std + (1 - sg) * other.std)
        self.count = max(self.count, other.count)
        return self
