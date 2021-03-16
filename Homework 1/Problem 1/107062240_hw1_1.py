import numpy as np

GIRD_WORLD_SHAPE = [4, 4]

class Transition:
    def __init__(self, next_state_id):
        self.probability = 0.25
        self.next_state = next_state_id
        self.reward = 0
        self.done = False

class Policy:
    def __init__(self, id):
        self.id = id
        self.transitions = []
        self.initial_transitions()

    def initial_transitions(self):
        if self.id == 0 or self.id == 15:   # terminal state
            self.transitions.append(Transition(self.id))
        else:
            index_op = [-4, 1, 4, -1]       # UP, RIGHT, DOWN, LEFT
            for op in index_op:
                next_state_id = self.id + op
                if next_state_id >= 0 and next_state_id <= 15:
                    self.transitions.append(Transition(next_state_id))

    def show_transitions(self):
        print("current state id: {}".format(self.id))
        for t in self.transitions:
            print("next state id: {}".format(t.next_state))


def policy_iteration(policy, discount_factor = 0.9, theta = 0.000001):
    
    V = np.zeros(GIRD_WORLD_SHAPE)

    while True:

        delta = 0

    return np.array(V)


# new_policy = Policy(0)
# new_policy.show_transitions()

# nn_policy = Policy(9)
# nn_policy.show_transitions()