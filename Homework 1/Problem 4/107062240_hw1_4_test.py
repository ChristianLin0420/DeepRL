from sys import stdin

import numpy as np
import pickle

BOARD_ROWS = 4
BOARD_COLS = 4
BOARD_DEPTH = 4

class State:
    def __init__(self, p1, current_board, symbol):
        self.board = current_board
        self.p1 = p1
        self.isEnd = False
        self.boardHash = None
        # init p1 plays first
        self.playerSymbol = symbol

    def updateState(self, position):
        self.board[position] = self.playerSymbol
        # switch to another player
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    # play with human
    def get_next_step(self):
        positions = self.availablePositions()
        p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
        print(p1_action)
        # take action and upate board state
        self.updateState(p1_action)
        # self.showBoard()

    def availablePositions(self):
        positions = []
        for d in range(BOARD_DEPTH):
            for i in range(BOARD_ROWS):
                for j in range(BOARD_COLS):
                    if self.board[d, i, j] == 0:
                        positions.append((d, i, j))  # need to be tuple
        return positions           

    def showBoard(self):
        # p1: x  p2: o
        for i in range(0, BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = 'x'
                if self.board[i, j] == -1:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')        

class Player:
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.states = []  # record all positions taken
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}  # state -> value

    def getHash(self, board):
        boardHash = str(board.reshape(BOARD_COLS * BOARD_ROWS * BOARD_DEPTH))
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

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()

if __name__ == "__main__":

    # load pre-trained policy
    p1 = Player("computer", exp_rate=0)
    p1.loadPolicy("hw1_4_data")
    
    # Using readlines()
    # file1 = open('hw1-3_sample_input', 'r')
    # Lines = file1.readlines()
    
    count = 0
    # Strips the newline character

    for line in stdin:
        count += 1
        line = line.replace("\n", "")
        arr = line.split(' ')
        original = [int(numeric_string) for numeric_string in arr]
        original = np.array(original)
        desired_array = original[1:65].reshape([BOARD_DEPTH, BOARD_ROWS, BOARD_COLS])
        st = State(p1, desired_array, original[0])
        st.get_next_step()