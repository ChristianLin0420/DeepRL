import numpy as np

GRID_WORLD_SHAPE = [4, 4]
GRID_WORLD_COUNT = GRID_WORLD_SHAPE[0] * GRID_WORLD_SHAPE[1]
DISCOUNT_FACTOR = 1.0
THETA = 0.0001

class Transition:
    def __init__(self, next_state_id):
        self.probability = 0.25
        self.next_state = next_state_id
        self.reward = -1
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
                    if (op == 1 and next_state_id % 4 == 0) or (op == -1 and next_state_id % 4 == 3):
                        self.transitions.append(Transition(self.id))
                    else:
                        self.transitions.append(Transition(next_state_id))
                else:
                    self.transitions.append(Transition(self.id))

class Gridworld:
    def __init__(self):
        self.gridworld = []
        self.initial_gridworld()

    def initial_gridworld(self):
        for i in range(GRID_WORLD_COUNT):
            new_policy = Policy(i)
            self.gridworld.append(new_policy)

    def policy_iteration(self):

        V = np.zeros(GRID_WORLD_COUNT)
        iteration_count = 0
        
        while True:
            delta = 0
            # For each state, perform a "full backup"
            for s in range(GRID_WORLD_COUNT):

                if s == 0 or s == 15:
                    continue

                iteration_count += 1
                v = 0
                # Look at the possible next actions
                for transition in self.gridworld[s].transitions:
                    v += transition.probability * (transition.reward + DISCOUNT_FACTOR * V[transition.next_state])
                
                # How much our value function changed (across any states)
                delta = max(delta, np.abs(v - V[s]))
                V[s] = v
            
            if delta < THETA:
                self.show_states(V, iteration_count)
                break

    def show_states(self, V, iteration_count):
        print("iteration count: {}".format(iteration_count))

        for i in range(0, GRID_WORLD_SHAPE[0]):
            print('----------------------------------')
            out = '| '
            for j in range(0, GRID_WORLD_SHAPE[1]):
                out += str(round(V[i * GRID_WORLD_SHAPE[0] + j], 5)).ljust(9) + ' | '
            print(out)
        print('----------------------------------')

########### start main iteration ###########
gridworld = Gridworld()
gridworld.policy_iteration()
