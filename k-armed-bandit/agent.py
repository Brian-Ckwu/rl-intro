import random
from memory import Memory

class Agent(object):
    def __init__(self, k: int):
        self.k = k
        self.memory = Memory(k)

    # ε-greedy approach
    def select_action(self, eps: float):
        greedy = True if random.random() > eps else False
        if greedy:
            action = self.memory.action_rewards.argmax()
        else:
            action = random.randrange(0, self.k)
            
        return action