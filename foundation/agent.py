# %matplotlib inline

import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

from foundation.MCTS import MCTS, Stats, Node, Edge
from foundation.igame import IGameState
from utils import config, loggers as lg

from IPython import display


class User:
    def __init__(self, name, state_size, action_size):
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
    
    def act(self, state, tau):
        action = input('Enter your chosen action: ')
        pi = np.zeros(self.action_size)
        pi[action] = 1
        value = None
        NN_value = None
        return action, pi, value, NN_value


class Agent:
    def __init__(self, name, state_size, action_size, mcts_simulations, cpuct, model):
        self.name = name
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.cpuct = cpuct
        
        self.MCTSsimulations = mcts_simulations
        self.model = model
        
        self.mcts = None
        
        self.train_overall_loss = []
        self.train_value_loss = []
        self.train_policy_loss = []
        self.val_overall_loss = []
        self.val_value_loss = []
        self.val_policy_loss = []
    
    def simulate(self):
        
        lg.logger_mcts.info('ROOT NODE...%s', self.mcts.root.state.id())
        self.mcts.root.state.render(lg.logger_mcts)
        lg.logger_mcts.info('CURRENT PLAYER...%d', self.mcts.root.state.playerTurn)
        
        # ### MOVE THE LEAF NODE
        leaf, breadcrumbs, lastState = self.mcts.moveToLeaf()
        leaf.state.render(lg.logger_mcts)
        
        # ### EVALUATE THE LEAF NODE
        value = self.evaluateLeaf(leaf, lastState)
        
        # ### BACKFILL THE VALUE THROUGH THE TREE
        self.mcts.backFill(leaf, value, breadcrumbs)
    
    def act(self, state, tau):
        
        if self.mcts is None or state.id() not in self.mcts.idToNode:
            self.buildMCTS(state)
        else:
            self.changeRootMCTS(state)
        
        # ### run the simulation
        for sim in range(self.MCTSsimulations):
            lg.logger_mcts.info('***************************')
            lg.logger_mcts.info('****** SIMULATION %d ******', sim + 1)
            lg.logger_mcts.info('***************************')
            self.simulate()
        
        # ### get action values
        actionValues, values = self.getActionValue(1)
        
        # ###pick the action
        action, value = self.chooseAction(actionValues, values, tau)
        
        nextState = state.takeAction(action)
        
        NN_value = -self.get_preds(nextState)[0]
        
        lg.logger_mcts.info('ACTION VALUES...%s', actionValues)
        lg.logger_mcts.info('CHOSEN ACTION...%d', action)
        lg.logger_mcts.info('MCTS PERCEIVED VALUE...%f', value)
        lg.logger_mcts.info('NN PERCEIVED VALUE...%f', NN_value)
        
        return action, actionValues, value, NN_value
    
    def get_preds(self, state):
        # predict the leaf
        inputToModel = np.array([self.model.convertToModelInput(state)])
        
        prediction = self.model.predict(inputToModel)
        value_array = prediction[0]
        logits_array = prediction[1]
        value = value_array[0]
        
        logits = logits_array[0]
        
        allowedActions = state.allowedActions()
        
        mask = np.ones(logits.shape, dtype=bool)
        mask[allowedActions] = False
        logits[mask] = -100
        
        # SOFTMAX
        odds = np.exp(logits)
        probs = odds / np.sum(odds)  # ##put this just before the for?
        
        return value, probs, allowedActions
    
    def evaluateLeaf(self, leaf: Node, lastState: IGameState):
        
        lg.logger_mcts.info('------EVALUATING LEAF------')
        inProgress = lastState is None or not lastState.isEndGame()
        if inProgress:
            
            value, probs, allowedActions = self.get_preds(leaf.state)
            lg.logger_mcts.info('PREDICTED VALUE FOR %d: %f', leaf.state.playerTurn, value)
           
            for action in allowedActions:
                newState = leaf.state.takeAction(action)
                if newState.id() not in self.mcts.idToNode:
                    node = Node(newState)
                    self.mcts.addNode(node)
                    lg.logger_mcts.info('added node...%s...p = %f', node.id, probs[action])

                node = self.mcts.idToNode[newState.id()]
                newEdge = Edge(leaf, node, probs[action], action)
                leaf.add(action, newEdge)
        else:
            value = 0 if lastState is None else lastState.getWinner()
            lg.logger_mcts.info('GAME VALUE FOR %d: %f', leaf.playerTurn, value)
        
        return value
    
    def getActionValue(self, tau) -> ([int], [float]):
        edges = self.mcts.root.edges
        actionValues = np.zeros(self.action_size, dtype=np.integer)
        values = np.zeros(self.action_size, dtype=np.float32)
        
        for action, edge in edges:
            actionValues[action] = pow(edge[Stats.N], 1 / tau)
            values[action] = edge[Stats.Q]

        actionValues = actionValues / (np.sum(actionValues) * 1.0)
        return actionValues, values
    
    def chooseAction(self, actionValues: [int], values: [float], tau: int):
        if tau == 0:
            actions = np.argwhere(actionValues == max(actionValues))
            action = random.choice(actions)[0]
        else:
            action_idx = np.random.multinomial(1, actionValues)
            action = np.where(action_idx == 1)[0][0]
        
        value = values[action]
        
        return action, value
    
    def replay(self, memory):
        lg.logger_mcts.info('******RETRAINING MODEL******')
        
        for i in range(config.TRAINING_LOOPS):
            minibatch = memory.miniBach()
            
            training_states = np.array([self.model.convertToModelInput(row['state']) for row in minibatch])
            training_targets = {'value_head': np.array([row['value'] for row in minibatch]),
                                'policy_head': np.array([row['AV'] for row in minibatch])}
            
            fit = self.model.fit(training_states, training_targets,
                                 epochs=config.EPOCHS, verbose=1, validation_split=0, batch_size=32)
            
            lg.logger_mcts.info('NEW LOSS %s', fit.history)
            
            self.train_overall_loss.append(round(fit.history['loss'][config.EPOCHS - 1], 4))
            self.train_value_loss.append(round(fit.history['value_head_loss'][config.EPOCHS - 1], 4))
            self.train_policy_loss.append(round(fit.history['policy_head_loss'][config.EPOCHS - 1], 4))
        
        plt.plot(self.train_overall_loss, 'k')
        plt.plot(self.train_value_loss, 'k:')
        plt.plot(self.train_policy_loss, 'k--')
        
        plt.legend(['train_overall_loss', 'train_value_loss', 'train_policy_loss'], loc='lower left')
        
        display.clear_output(wait=True)
        display.display(pl.gcf())
        pl.gcf().clear()
        time.sleep(1.0)
        
        print('\n')
        self.model.printWeightAverages()
    
    def predict(self, inputToModel):
        preds = self.model.predict(inputToModel)
        return preds
    
    def buildMCTS(self, state):
        lg.logger_mcts.info('****** BUILDING NEW MCTS TREE FOR AGENT %s ******', self.name)
        self.root = Node(state)
        self.mcts = MCTS(self.root, self.cpuct)
    
    def changeRootMCTS(self, state):
        lg.logger_mcts.info('****** CHANGING ROOT OF MCTS TREE TO %s FOR AGENT %s ******', state.id(), self.name)
        self.mcts.root = self.mcts.idToNode[state.id()]
        
    
