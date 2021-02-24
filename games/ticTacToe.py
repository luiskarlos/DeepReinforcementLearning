import numpy as np
from methodtools import lru_cache

from foundation.igame import IGame, IGameState


class TTTGame(IGame):
    """
    Thanks to https://github.com/archcra/tic-tac-toe-az/blob/master/game.py
    """
    pieces = {'1': 'X', '0': '-', '-1': 'O'}
    
    def __init__(self):
        self.name = 'tictactoe'
        self.currentPlayer = 1
        self.gameState = TTTGameState(np.zeros(3 * 3, dtype=np.int), 1)
        self.actionSpace = np.zeros(3 * 3, dtype=np.int)
        self.grid_shape = (3, 3)
        self.input_shape = (2, 3, 3)
        self.state_size = len(self.gameState.binary())
        self.action_size = len(self.actionSpace)
    
    def reset(self):
        self.gameState = TTTGameState(np.zeros(3 * 3, dtype=np.int), 1)
        self.currentPlayer = 1
        return self.gameState
    
    def step(self, action):
        next_state = self.gameState.takeAction(action)
        self.gameState = next_state
        self.currentPlayer = -self.currentPlayer
        info = None
        return next_state, info
    
    def identities(self, state, actionValues):
        identities = [(state, actionValues)]
        return identities


class TTTGameState(IGameState):
    winners = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]
    ]
    pieces = TTTGame.pieces
    
    def __init__(self, board, playerTurn):
        self.board = board
        self.playerTurn = playerTurn

    @lru_cache(maxsize=128)
    def allowedActions(self):
        allowed = [i for i, e in enumerate(self.board) if e == 0]
        return allowed

    @lru_cache(maxsize=128)
    def binary(self):
        currentplayer_position = np.zeros(len(self.board), dtype=np.int)
        currentplayer_position[self.board == self.playerTurn] = 1
        
        other_position = np.zeros(len(self.board), dtype=np.int)
        other_position[self.board == -self.playerTurn] = 1
        
        position = np.append(currentplayer_position, other_position)
        
        return position

    @lru_cache(maxsize=128)
    def id(self):
        player1_position = np.zeros(len(self.board), dtype=np.int)
        player1_position[self.board == 1] = 1
        
        other_position = np.zeros(len(self.board), dtype=np.int)
        other_position[self.board == -1] = 1
        
        position = np.append(player1_position, other_position)
        
        id = ''.join(map(str, position))
        
        return id

    @lru_cache(maxsize=128)
    def isEndGame(self):
        if np.count_nonzero(self.board) == 9:
            return 1
        
        for x, y, z in self.winners:
            if self.board[x] + self.board[y] + self.board[z] == 3 * -self.playerTurn:
                return 1
        return 0

    @lru_cache(maxsize=128)
    def values(self):
        # This is the value of the state for the current player
        # i.e. if the previous player played a winning move, you lose
        for x, y, z in self.winners:
            if self.board[x] + self.board[y] + self.board[z] == 3 * -self.playerTurn:
                return -self.playerTurn, -1, 1
        return 0, 0, 0
    
    def score(self):
        tmp = self.values()
        return tmp[1], tmp[2]
    
    def takeAction(self, action):
        newBoard = np.array(self.board)
        newBoard[action] = self.playerTurn
        
        newState = TTTGameState(newBoard, -self.playerTurn)
        
        return newState
    
    def render(self, logger):
        for r in range(3):
            logger.info([self.pieces[str(x)] for x in self.board[3 * r: (3 * r + 3)]])
        logger.info('--------------')
        
    def getWinner(self):
        return self.values()[0]

