import numpy as np

GIRD_WORLD_SHAPE = [4, 4]

class Transition:
    def __init__(self, next_state_id, reward, done):
        self.probability = 0.25
        self.next_state = next_state_id
        self.reward = reward
        self.done = done

class Policy:
    def __init__(self, id):
        self.id = id




def policy_iteration(policy, discount_factor = 0.9, theta = 0.000001):
    
    V = np.zeros(GIRD_WORLD_SHAPE)

    while True:
        
        delta = 0



    return np.array(V)
