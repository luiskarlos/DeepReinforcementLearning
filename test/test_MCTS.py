from unittest import TestCase
from unittest.mock import patch

from foundation.MCTS import Node, MCTS, Stats, Edge


class TestMCTS(TestCase):

    @patch('foundation.igame.IGameState')
    def test(self, gameState):
        CPUCT = 1 # defined in config
        gameState.playerTurn = 1
        node = Node(gameState)
        mcts = MCTS(node, CPUCT)
        currentNode, value, done, breadcrumbs = mcts.moveToLeaf()
        self.assertEqual(currentNode.playerTurn, 1)

    @patch('foundation.igame.IGameState')
    def test_node_Nb(self, gameState):
        node = Node(gameState)
        inNode = Node(gameState)
        outNode = Node(gameState)
        node.add(0, Edge(inNode, outNode, 1, 2))
        node.add(1, Edge(inNode, outNode, 1, 3))

        node.edges[0][1][Stats.N] = 10
        node.edges[1][1][Stats.N] = 10

        nb = node._Nb()
        self.assertEqual(20, nb)

    @patch('foundation.igame.IGameState')
    @patch("foundation.MCTS.np.random.dirichlet")
    def test_find_move(self, dirichlet, gameState):
        dirichlet.return_value = [0.1, 0.5]
        node = Node(gameState)
        inNode = Node(gameState)
        outNode = Node(gameState)
        node.add(0, Edge(inNode, outNode, 1, 2))
        node.add(1, Edge(inNode, outNode, 1, 3))

        node.edges[0][1][Stats.N] = 10
        node.edges[1][1][Stats.N] = 10

        simulationAction, simulationEdge = node.findMove(1, True)
        self.assertEqual(1, simulationAction)

