import numpy as np
import torch
from torch.distributions import Normal

class Levers(object):
    def __init__(self, k: int):
        norm_dist = Normal(loc=0.0, scale=1.0)
        self.mean_rewards = norm_dist.sample((k,)).numpy()
        self.lever_dists = [Normal(loc=mu, scale=1.0) for mu in self.mean_rewards]
    
    def pull(self, idx: int):
        return self.lever_dists[idx].sample().item()