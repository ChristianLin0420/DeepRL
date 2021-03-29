import numpy as np
import pickle

BOARD_ROWS = 4
BOARD_COLS = 4
BOARD_DEPTH = 4

class State:
    def __init__(self, p1, p2):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS, BOARD_DEPTH))
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        self.playerSymbol = 1

    # get unique hash of current board state
    def getHash(self):
        self.boardHash = str(self.board.reshape(BOARD_ROWS * BOARD_COLS * BOARD_DEPTH))
        return self.boardHash

    def winner(self):
        # row
        for i in range(BOARD_ROWS):
            for j in range(BOARD_DEPTH):
                if sum(self.board[i, : , j]) == 4:
                    self.isEnd = True
                    return 1
                if sum(self.board[i, : , j]) == -4:
                    self.isEnd = True
                    return -1
        
        # col
        for i in range(BOARD_COLS):
            for j in range(BOARD_DEPTH):
                if sum(self.board[: , i, j]) == 4:
                    self.isEnd = True
                    return 1
                if sum(self.board[:, i, j]) == -4:
                    self.isEnd = True
                    return -1
                
        # depth
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if sum(self.board[i, j, :]) == 4:
                    self.isEnd = True
                    return 1
                if sum(self.board[i, j, :]) == -4:
                    self.isEnd = True
                    return -1

        # diagonal (2D)
        for i in range(BOARD_ROWS):
            diag_sum1 = 0
            diag_sum2 = 0
            for j in range(BOARD_DEPTH):
                diag_sum1 += self.board[i, i, j]
                diag_sum2 += self.board[BOARD_ROWS - i - 1, i , j]
            d_sum = max(abs(diag_sum1), abs(diag_sum2))
            if d_sum == 4:
                self.isEnd = True
                return 1 if diag_sum1 == 4 else -1

        for i in range(BOARD_COLS):
            diag_sum1 = 0
            diag_sum2 = 0
            for j in range(BOARD_DEPTH):
                diag_sum1 += self.board[j, i, i]
                diag_sum2 += self.board[j, i, BOARD_DEPTH - i - 1]
            d_sum = max(abs(diag_sum1), abs(diag_sum2))
            if d_sum == 4:
                self.isEnd = True
                return 1 if diag_sum1 == 4 else -1

        for i in range(BOARD_ROWS):
            diag_sum1 = 0
            diag_sum2 = 0
            for j in range(BOARD_COLS):
                diag_sum1 += self.board[i, j, i]
                diag_sum2 += self.board[BOARD_ROWS - i - 1, i, i]
            d_sum = max(abs(diag_sum1), abs(diag_sum2))
            if d_sum == 4:
                self.isEnd = True
                return 1 if diag_sum1 == 4 else -1

        # tie
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0

        self.isEnd = False
        return None

    def availablePositions(self):
        positions = []

        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                for k in range(BOARD_DEPTH):
                    if self.board[i, j, k] == 0:
                        positions.append((i, j, k))
    
        return positions

    def updateState(self, position):
        self.board[position] = self.playerSymbol
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    # only when game ends
    def giveReward(self):
        result = self.winner()
        # backpropagate reward
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.1)
            self.p2.feedReward(0.5)

    # board reset
    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS, BOARD_DEPTH))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

    def play(self, rounds=100):
        for i in range(rounds):
            if i % 10 == 0:
                print("Rounds {}".format(i))
            while not self.isEnd:
                # Player 1
                positions = self.availablePositions()
                p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                # take action and upate board state
                self.updateState(p1_action)
                board_hash = self.getHash()
                self.p1.addState(board_hash)
                # check board status if it is end

                win = self.winner()
                # print("player 1 win state: {}".format(win))
                if win is not None:
                    # ended with p1 either win or draw
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break
                else:
                    # Player 2
                    positions = self.availablePositions()
                    p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                    self.updateState(p2_action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)

                    win = self.winner()
                    # print("player 2 win state: {}".format(win))
                    if win is not None:
                        # ended with p2 either win or draw
                        self.giveReward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break

        pp1 = self.p1.states_value
        pp2 = self.p2.states_value
        self.p1.states_value.update(pp2)
        self.p2.states_value.update(pp1)
        self.p1.savePolicy()
        self.p2.savePolicy()

    # play with human
    def play2(self):
        while not self.isEnd:
            # Player 1
            positions = self.availablePositions()
            p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
            # take action and upate board state
            self.updateState(p1_action)
            self.showBoard()
            # check board status if it is end
            win = self.winner()
            if win is not None:
                if win == 1:
                    print(self.p1.name, "wins!")
                else:
                    print("tie!")
                self.reset()
                break

            else:
                # Player 2
                positions = self.availablePositions()
                p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)

                self.updateState(p2_action)
                self.showBoard()
                win = self.winner()
                if win is not None:
                    if win == -1:
                        print(self.p2.name, "wins!")
                    else:
                        print("tie!")
                    self.reset()
                    break

    def showBoard(self):
        for i in range(BOARD_DEPTH):
            print('=================')
            for j in range(BOARD_ROWS):
                print('-------------')
                out = '| '
                for k in range(0, BOARD_COLS):
                    if self.board[i, j, k] == 1:
                        token = 'x'
                    if self.board[i, j, k] == -1:
                        token = 'o'
                    if self.board[i, j, k] == 0:
                        token = ' '
                    out += token + ' | '
                print(out)


class Player:
    def __init__(self, name, exp_rate = 0.3):
        self.name = name
        self.states = []
        self.learning_rate = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}

    def getHash(self, board):
        boardHash = str(board.reshape(BOARD_ROWS * BOARD_COLS * BOARD_DEPTH))
        return boardHash

    def chooseAction(self, positions, current_board, symbol):
        if np.random.uniform(0, 1) <= self.exp_rate:
            # take random action
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            value_max = -999
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_boardHash = self.getHash(next_board)
                value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
                # print("value", value)
                if value >= value_max:
                    value_max = value
                    action = p
        # print("{} takes action {}".format(self.name, action))
        return action
    
    # append a hash state
    def addState(self, state):
        self.states.append(state)

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.learning_rate * (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]

    def reset(self):
        self.states = []

    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()

class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions):
        while True:
            depth = int(input("Input your action depth:"))
            row = int(input("Input your action row:"))
            col = int(input("Input your action col:"))
            action = (depth, row, col)
            if action in positions:
                return action

if __name__ == "__main__":
    # training
    p1 = Player("p1")
    p2 = Player("p2")

    st = State(p1, p2)
    print("training...")
    st.play(20000)

    # play with human
    p1 = Player("computer", exp_rate=0)
    p1.loadPolicy("policy_p1")

    p2 = HumanPlayer("human")
    # p2 = Player("computer", exp_rate=0)
    # p2.loadPolicy("policy_p2")

    st = State(p1, p2)
    st.play2()