import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle

BOARD_ROWS = 4
BOARD_COLS = 4
BOARD_DEPTH = 4

EXPLORATION_CONSTANT = 2.5

class Node:
    def __init__(self, player, state, parent = None):
        self.state = state
        self.player = player
        self.childs = []
        self.parent = parent
        self.visited_time = 0
        self.cululative_reward = 0
        self.q = None

class MonteCarloSearchTree:

    def __init__(self, max_depth, iteration_count, player):
        self.id_count = 0
        self.max_depth = max_depth
        self.iteration_count = iteration_count
        self.world = np.zeros([BOARD_DEPTH, BOARD_ROWS, BOARD_COLS])
        self.total_node_count = 0
        self.tree = self.initial_game(player)

        # print(self.world.shape) 

    def initial_game(self, player):
        tree = {str(self.id_count) : Node(player, self.world)}
        self.id_count += 1
        return tree

    def selection(self):

        leaf_node_found = False
        leaf_node_id = 0

        while not leaf_node_found:
            node_id = leaf_node_id
            n_child = len(self.tree[str(node_id)].childs)

            print("node id: ", node_id)

            if n_child == 0:
                leaf_node_id = node_id
                leaf_node_found = True
            else:
                maximum_uct_value = -100.0

                for i in range(n_child):
                    action = self.tree[str(node_id)].childs[i]

                    child_id = str(node_id) + str(action)
                    w = self.tree[child_id].w
                    n = self.tree[child_id].n

                    total_n = self.total_node_count

                    if n == 0:
                        n = 1e-4
                    
                    exploitation_value = w / n
                    exploration_value = np.sqrt(np.log(total_n / n))
                    uct_value = exploitation_value + self.EXPLORATION_CONSTANT * exploitation_value

                    if uct_value > maximum_uct_value:
                        maximum_uct_value = uct_value
                        leaf_node_id = child_id

        depth = len(str(leaf_node_id))

        return str(leaf_node_id), depth

    def expansion(self, leaf_node_id):
        
        current_board = self.tree[leaf_node_id].state
        winner = self.terminal(current_board)
        possible_actions = self.get_valid_actions(current_board)

        while winner is None:

            childs = []

            for action_set in possible_actions:
                action, action_index = action_set
                state = self.tree[leaf_node_id].state
                current_player = self.tree[leaf_node_id].player

                next_trun = -current_player
                state[action] = current_player

                child_id = leaf_node_id + str(action_index)
                childs.append(child_id)

                self.tree[child_id] = Node(next_trun, state, leaf_node_id)
                self.id_count += 1
                self.tree[leaf_node_id].childs.append(action_index)

            random_index = np.random.randint(low = 0, high = len(childs), size = 1)
            child_node_id = childs[random_index[0]]

        return child_node_id

    def terminal(self, current_board):

        def who_wins(sums):
            if np.any(sums == CONSECUTIVRE_NODES_COUNT):
                return 1
            if np.any(sums == -CONSECUTIVRE_NODES_COUNT):
                return -1
            return None

        def terminal_in_conv(current_board):
            # row
            for i in range(BOARD_ROWS):
                for j in range(BOARD_DEPTH):
                    sums = sum(leaf_state[i, :, j])
                    result = who_wins(sums)

                    if result is not None:
                        return result
            
            # col
            for i in range(BOARD_COLS):
                for j in range(BOARD_DEPTH):
                    sums = sum(leaf_state[: , i, j])
                    result = who_wins(sums)

                    if result is not None:
                        return result
                    
            # depth
            for i in range(BOARD_ROWS):
                for j in range(BOARD_COLS):
                    sums = sum(leaf_state[i, j, :])
                    result = who_wins(sums)

                    if result is not None:
                        return result

            # diagonal (2D)
            for i in range(BOARD_ROWS):
                diag_sum1 = 0
                diag_sum2 = 0
                for j in range(BOARD_DEPTH):
                    diag_sum1 += leaf_state[i, i, j]
                    diag_sum2 += leaf_state[BOARD_ROWS - i - 1, i , j]
                d_sum = max(abs(diag_sum1), abs(diag_sum2))
                if d_sum == 4:
                    return 1 if diag_sum1 == 4 else -1

            for i in range(BOARD_COLS):
                diag_sum1 = 0
                diag_sum2 = 0
                for j in range(BOARD_DEPTH):
                    diag_sum1 += leaf_state[j, i, i]
                    diag_sum2 += leaf_state[j, i, BOARD_DEPTH - i - 1]
                d_sum = max(abs(diag_sum1), abs(diag_sum2))
                if d_sum == 4:
                    return 1 if diag_sum1 == 4 else -1

            for i in range(BOARD_ROWS):
                diag_sum1 = 0
                diag_sum2 = 0
                for j in range(BOARD_COLS):
                    diag_sum1 += leaf_state[i, j, i]
                    diag_sum2 += leaf_state[BOARD_ROWS - i - 1, i, i]
                d_sum = max(abs(diag_sum1), abs(diag_sum2))
                if d_sum == 4:
                    return 1 if diag_sum1 == 4 else -1
            
            return None
        
        valid_space_count = len(self.get_valid_actions(current_board))

        if valid_space_count == 0:
            return 2 # no available place to play (tie)
            
        return None

    def get_valid_actions(self, current_board):

        actions = []
        count = 0

        for d in range(BOARD_DEPTH):
            for r in range(BOARD_ROWS):
                for c in range(BOARD_COLS):
                
                    if current_board[d, r, c] == 0:
                        actions.append([(d, r, c), count])

                count += 1

        return actions

    def simulation(self, child_node_id):

        self.total_node_count += 1
        state = self.tree[child_node_id].state
        previous_player = self.tree[child_node_id].player
        finish = False

        while not finish:

            winner = self.terminal(state)

            if winner is not None:
                finish = True
            else:
                possible_actions = self.get_valid_actions(state)
                random_index = np.random.randint(low = 0, high = len(possible_actions), size = 1)[0]
                action, _ = possible_actions[random_index]

                current_player = -previous_player
                state[action] = -previous_player

                previous_player = current_player

        return winner

    def back_propagate(self, child_node_id, winner):
        player = self.tree[child_node_id].player

        if winner == 2:
            reward = 0
        elif winner == player:
            reward = 1
        else:
            reward = -1

        finish_back_propagate = False
        node_id = child_node_id

        while not finish_back_propagate:
            self.tree[node_id].visited_time += 1
            self.tree[node_id].cululative_reward += reward
            self.tree[node_id].q = self.tree[node_id].visited_time / self.tree[node_id].cululative_reward
            parent_id = self.tree[node_id].parent

            if parent_id == '0':
                self.tree[parent_id].visited_time += 1
                self.tree[parent_id].cululative_reward += reward
                self.tree[parent_id].q = self.tree[parent_id].visited_time / self.tree[parent_id].cululative_reward
                finish_back_propagate = True
            else:
                node_id = parent_id
    
    def solve(self):
        for i in range(self.iteration_count):
            leaf_node_id, depth_searched = self.selection()
            child_node_id = self.expansion(leaf_node_id)
            winner = self.simulation(child_node_id)
            self.backprop(child_node_id, winner)

            if depth_searched > self.depth:
                break

        # SELECT BEST ACTION
        current_state_node_id = '0'
        action_candidates = self.tree[current_state_node_id].childs

        best_q = -100

        for a in action_candidates:
            q = self.tree[(0,)+(a,)]['q']
            if q > best_q:
                best_q = q
                best_action = a

        return best_action, best_q, depth_searched

    def savePolicy(self):
        fw = open('MCST_policy', 'wb')
        pickle.dump(self.tree, fw)
        fw.close()

if __name__ == '__main__':
    mcts = MonteCarloSearchTree(4, 1, 1)
    mcts.savePolicy()
    best_action, max_q = mcts.solve()
    print('best action= ', best_action, ' max_q= ', max_q)