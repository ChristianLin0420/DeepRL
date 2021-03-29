import numpy as np
import matplotlib.pyplot as plt

ITERATION_COUNT = 100
TREE_DEPTH = 10
EXPLORATION_CONSTANT = 2.5
CONSECUTIVRE_NODES_COUNT = 4

class policy():
    def __init__(self):
        self.tree = {}

class MonteCarloSearchTree():
    def __init__(self, tree = None, world = None, player = None):
        self.total_n = 0
        self.leaf_node_id = None
        
        if tree == None:
            self.tree = self.initial_game(world, player)
        else:
            self.tree = tree

    def initial_game(self, board, player):
        root_id = (0, )
        
        tree = {root_id : {
            'state'  : board,
            'player' : player,
            'child' : [],
            'parent' : None,
            'n'      : 0,
            'w'      : 0, 
            'q'      : None
        }}

        return tree

    '''
    select leaf node which have maximum uct value
    '''
    def select(self):

        leaf_node_found = False
        leaf_node_id = (0, )

        while not leaf_node_found:
            node_id = leaf_node_id
            n_child = len(self.tree[node_id]['child'])

            if n_child == 0:
                leaf_node_id = node_id
                leaf_node_found = True
            else:
                maximum_uct_value = -100.0

                for i in range(n_child):
                    action = self.tree[node_id]['child'][i]

                    child_id = node_id + (action, )
                    w = self.tree[child_id]['w']
                    n = self.tree[child_id]['n']

                    total_n = self.total_n

                    if n == 0:
                        n = 1e-4
                    
                    exploitation_value = w / n
                    exploration_value = np.sqrt(np.log(total_n / n))
                    uct_value = exploitation_value + self.EXPLORATION_CONSTANT * exploitation_value

                    if uct_value > maximum_uct_value:
                        maximum_uct_value = uct_value
                        leaf_node_id = child_id

        depth = len(leaf_node_id)

        return leaf_node_id, depth

    
    '''
    create all possible outcomes from leaf node
    '''
    def expansion(self, leaf_node_id):

        leaf_state = self.tree[leaf_node_id]['state']
        winner = self.terminal(leaf_state)
        possible_actions = self.get_valid_actions(leaf_state)

        child_node_id = leaf_node_id

        if winner is None:
            childs = []

            for action_set in posiible_actions:
                action, action_index = action_set
                state = self.tree[leaf_node_id]['state']
                current_player = self.tree[leaf_node_id]['player']

                if current_player == 'o':
                    next_turn = 'x'
                    state[action] = 1
                else:
                    next_turn = 'o'
                    state[action] = -1

                child_id = leaf_node_id + (action_index, )
                childs.append(child_id)
                self.tree[child_id] = {
                    'state'  : board,
                    'player' : next_turn,
                    'child'  : [],
                    'parent' : leaf_node_id,
                    'n'      : 0,
                    'w'      : 0, 
                    'q'      : None
                }
                self.tree[leaf_node_id]['child'].append(action_index)
            
            random_index = np.random.randint(low = 0, high = len(childs), size = 1)
            child_node_id = childs[random_index[0]]
        return child_node_id
    

    def terminal(self, leaf_state):

        def who_wins(sums):
            if np.any(sums == CONSECUTIVRE_NODES_COUNT):
                return 'o'
            if np.any(sums == -CONSECUTIVRE_NODES_COUNT):
                return 'x'
            return None

        def terminal_in_conv(leaf_state):
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
                    return 'o' if diag_sum1 == 4 else 'x'

            for i in range(BOARD_COLS):
                diag_sum1 = 0
                diag_sum2 = 0
                for j in range(BOARD_DEPTH):
                    diag_sum1 += leaf_state[j, i, i]
                    diag_sum2 += leaf_state[j, i, BOARD_DEPTH - i - 1]
                d_sum = max(abs(diag_sum1), abs(diag_sum2))
                if d_sum == 4:
                    return 'o' if diag_sum1 == 4 else 'x'

            for i in range(BOARD_ROWS):
                diag_sum1 = 0
                diag_sum2 = 0
                for j in range(BOARD_COLS):
                    diag_sum1 += leaf_state[i, j, i]
                    diag_sum2 += leaf_state[BOARD_ROWS - i - 1, i, i]
                d_sum = max(abs(diag_sum1), abs(diag_sum2))
                if d_sum == 4:
                    return 'o' if diag_sum1 == 4 else 'x'
            
            return None

        n_rows_board = len(self.tree[(0, )]['state'])
        window_size = CONSECUTIVRE_NODES_COUNT
        window_positions = range(n_rows_board - CONSECUTIVRE_NODES_COUNT + 1)

        for d in window_positions:
            for row in window_positions:
                for col in window_positions:
                    window = leaf_state[d:d + window_size, row:row + window_size, col: col + window_size]
                    winner = terminal_in_conv(window)

                    if winner is not None:
                        return winner

        if not np.any(leaf_state == 0):
            return 'draw'

        return None
            



    def get_valid_actions(self, leaf_state):

        actions = []
        count = 0
        state_size = len(leaf_state)

        for i in range(state_size):
            for j in range(state_size):
                if leaf_state[i][j] == 0:
                    actions.append([(i, j), count])
                count += 1

        return actions

    def simulation(self, child_node_id):

        self.total_n += 1
        state = self.tree[child_node_id]['state']
        previous_player = self.tree[child_node_id]['player']
        finish = False

        while not finish:
            winner = self.terminal(state)

            if winner is not None:
                finish = True
            else:
                possible_actions = self.get_valid_actions(state)
                random_index = np.random.randint(low = 0, high = len(possible_actions), size = 1)[0]
                action, _ = possible_actions[random_index]

                if previous_player == 'o':
                    current_player = 'x'
                    state[action] = -1
                else:
                    current_player = 'o'
                    state[action] = 1

                previous_player = current_player
        
        return winner

    def back_propagate(self, child_node_id, winner):
        player = self.tree[child_node_id]['palyer']

        if winner == 'draw':
            reward = 0
        elif winner == player:
            reward = 1
        else:
            reward = -1

        finish_back_propage = False
        node_id = child_node_id

        while not finish_back_propage:
            self.tree[node_id]['n'] += 1
            self.tree[node_id]['w'] += reward
            self.tree[node_id]['q'] = self.tree[node_id]['w'] / self.tree[node_id]['n']
            parent_id = self.tree[node_id]['parent']

            if parent_id == (0, ):
                self.tree[parent_id]['n'] += 1
                self.tree[parent_id]['w'] += reward
                self.tree[parent_id]['q'] = self.tree[parent_id]['w'] / self.tree[parent_id]['n']
                finish_back_propage = True
            else:
                node_id = parent_id

    def solve(self):

        for i in range(ITERATION_COUNT):
            left_node_id, depth_search = self.select()
            child_node_id = self.expansion(left_node_id)
            winner = self.simulation(child_node_id)
            self.back_propagate(child_node_id, winner)

            if depth_search > TREE_DEPTH:
                break

        current_state_node_id = (0, )
        action_candidates = self.tree[current_state_node_id]['child']
        best_q = -100

        for a in action_candidates:
            q = self.tree[(0, ) + (a, )]['q']

            if q > best_q:
                best_q = q
                best_action = action

        # FOR DEBUGGING
        print('\n----------------------')
        print(' [-] game board: ')
        for row in self.tree[(0,)]['state']:
            print (row)
        print(' [-] person to play: ', self.tree[(0,)]['player'])
        print('\n [-] best_action: %d' % best_action)
        print(' best_q = %.2f' % (best_q))
        print(' [-] searching depth = %d' % (depth_searched))

        # FOR DEBUGGING
        fig = plt.figure(figsize=(5,5))
        for a in action_candidates:
            # print('a= ', a)
            _node = self.tree[(0,)+(a,)]
            _state = deepcopy(_node['state'])

            _q = _node['q']
            _action_onehot = np.zeros(len(_state)**2)

            plt.subplot(len(_state),len(_state),a+1)
            plt.pcolormesh(_state, alpha=0.7, cmap="RdBu")
            plt.axis('equal')
            plt.gca().invert_yaxis()
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title('[%d] q=%.2f' % (a,_q))

        plt.draw()
        plt.waitforbuttonpress(0)
        plt.close(fig)


        return best_action, best_q, depth_searched

if __name__ == '__main__':
    mcts = MonteCarloSearchTree(tree = None)

    best_action, max_q = mcts.solve()
    print('best action= ', best_action, ' max_q= ', max_q)