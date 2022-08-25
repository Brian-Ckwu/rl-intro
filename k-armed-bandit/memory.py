import numpy as np

class Memory(object):
    def __init__(self, k: int):
        self.action_rewards = np.zeros(k, dtype=np.float64)
        self.action_counts = np.zeros(k, dtype=np.int64)
        self.received_rewards = list()

    def update_estimates(self, lever: int, reward: float) -> None:
        old_estimate = self.action_rewards[lever]
        old_count = self.action_counts[lever]

        # update reward
        if old_count == 0:
            self.action_rewards[lever] = reward
        else:
            self.action_rewards[lever] = (old_estimate * old_count + reward) / (old_count + 1)

        # update count
        self.action_counts[lever] = old_count + 1