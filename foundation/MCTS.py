import numpy as np
from functional import seq
from enum import Enum, auto

from utils import config, loggers as lg


class Stats(Enum):
    N = (auto(), "The Number of times action has been taken from state S")
    W = (auto(), "The total value of the next state")
    Q = (auto(), "The Mean value of the next state")
    P = (auto(), "The prior probability of selecting action a")
    
    def describe(self):
        return self.value[1]


class Node:
    def __init__(self, state):
        self.state = state
        self.playerTurn = state.playerTurn  # TODO: lk - remove this
        self.id = state.id
        self.edges: list[(int, 'Edge')] = []  # tuples of (action, edge)
    
    def edgesCount(self) -> int:
        return len(self.edges)
    
    def add(self, action: int, edge: 'Edge'):
        self.edges.append((action, edge))
    
    def _Nb(self) -> float:
        return seq(self.edges) \
            .map(lambda _edge: _edge[1][Stats.N]) \
            .sum()
    
    def isLeaf(self) -> bool:
        return len(self.edges) <= 0
    
    def findMove(self, cpuct, isRoot) -> (int, 'Node'):
        maxQU = -99999
        if isRoot:
            epsilon = config.EPSILON
            nu = np.random.dirichlet([config.ALPHA] * self.edgesCount())
        else:
            epsilon = 0
            nu = [0] * self.edgesCount()
        
        sqrtNb = np.sqrt(self._Nb())
        
        simulationAction, simulationEdge = None, None
        
        action: int
        for idx, (action, edge) in enumerate(self.edges):
            # Early in the simulation, U dominates (more exploration)
            # Later Q is more important (less exploration)
            
            # U: A function of P and N that increases if an action hasn't been
            # explored much relative to other actions or if prior probability of the action is high
            # https://www.youtube.com/watch?v=NjeYgIbPMmg
            U = cpuct * \
                ((1 - epsilon) * edge[Stats.P] +
                 epsilon * nu[idx]) * sqrtNb / (1 + edge[Stats.N])
            
            Q = edge[Stats.Q]
            
            QU = Q + U
            
            lg.logger_mcts.info(
                'action: %d (%d)... N = %d, P = %f, nu = %f, adjP = %f, W = %f, Q = %f, U = %f, Q+U = %f',
                action, action % 7, edge[Stats.N], np.round(edge[Stats.P], 6), np.round(nu[idx], 6),
                ((1 - epsilon) * edge[Stats.P] + epsilon * nu[idx])
                , np.round(edge[Stats.W], 6), np.round(Q, 6), np.round(U, 6), np.round(QU, 6))
            
            if QU > maxQU:
                maxQU = QU
                simulationAction = action
                simulationEdge = edge
        
        return simulationAction, simulationEdge


class Edge:
    
    def __init__(self, inNode: Node, outNode: Node, prior, action: int):
        self.id = inNode.state.id + '|' + outNode.state.id
        self.inNode = inNode
        self.outNode = outNode
        self.playerTurn = inNode.state.playerTurn  # TODO: lk - remove this
        self.action = action
        
        self.stats = {
            Stats.N: 0,
            Stats.W: 0,
            Stats.Q: 0,
            Stats.P: prior,
        }
    
    def __getitem__(self, stat):
        return self.stats[stat]
    
    def __setitem__(self, stat, value):
        self.stats[stat] = value


class MCTS:
    """
        https://www.analyticsvidhya.com/blog/2019/01/monte-carlo-tree-search-introduction-algorithm-deepmind-alphago/
        The result is an “informed MCTS” which incorporates the outputs of the neural network to guide
        search.
    """
    
    def __init__(self, root: Node, cpuct: float):
        self.root = root  # tree root
        self.idToNode = {}  # map ID -> Node
        self.cpuct = cpuct  # see config
        
        self.addNode(root)
    
    def __len__(self):
        return len(self.idToNode)
    
    def moveToLeaf(self):
        lg.logger_mcts.info('------MOVING TO LEAF------')
        breadcrumbs = []
        currentNode = self.root
        
        done = 0
        value = 0
        
        while not currentNode.isLeaf():
            lg.logger_mcts.info('PLAYER TURN...%d', currentNode.state.playerTurn)
            
            simulationAction, simulationEdge = currentNode.findMove(self.cpuct, currentNode == self.root)
            
            lg.logger_mcts.info('action with highest Q + U...%d', simulationAction)
            
            newState, value, done = currentNode.state.takeAction(simulationAction)  # the value of the newState from the POV of the new playerTurn
            currentNode = simulationEdge.outNode
            breadcrumbs.append(simulationEdge)
        
        lg.logger_mcts.info('DONE...%d', done)
        
        return currentNode, value, done, breadcrumbs
    
    @staticmethod
    def backFill(leaf: Node, value, breadcrumbs):
        lg.logger_mcts.info('------DOING BACKFILL------')
        
        currentPlayer = leaf.state.playerTurn
        
        for edge in breadcrumbs:
            playerTurn = edge.playerTurn
            if playerTurn == currentPlayer:
                direction = 1
            else:
                direction = -1
            
            edge[Stats.N] = edge[Stats.N] + 1
            edge[Stats.W] = edge[Stats.W] + value * direction
            edge[Stats.Q] = edge[Stats.W] / edge[Stats.N]
            
            lg.logger_mcts.info('updating edge with value %f for player %d... N = %d, W = %f, Q = %f'
                                , value * direction
                                , playerTurn
                                , edge[Stats.N]
                                , edge[Stats.W]
                                , edge[Stats.Q]
                                )
            
            edge.outNode.state.render(lg.logger_mcts)
    
    def addNode(self, node: Node):
        self.idToNode[node.id] = node
