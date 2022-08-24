import numpy as np

class Memory(object):
    def __init__(self, k: int):
        self.action_rewards = np.zeros(k, dtype=np.float64)
        self.action_counts = np.zeros(k, dtype=np.int64)
