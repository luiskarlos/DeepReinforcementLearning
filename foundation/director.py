import numpy as np
import random
import pickle
from typing import Callable

from foundation.igame import IGame
from foundation.memory import Memory
from foundation.model import Residual_CNN, Gen_Model
from foundation.agent import Agent, User

from utils import loggers as lg, config


class Director:
    best_player_version: int
    
    def __init__(self, gameProvider=Callable[[], IGame]):
        self.gameProvider = gameProvider
        self.gameEnv: IGame = gameProvider()
        # create an untrained neural network objects from the config file
        self.best_player_version = 0
        self.current_NN = self._createCNN()
        self.best_NN = self._createCNN()
        self.current_player = self._createAgent('current_player', self.current_NN)
        self.best_player = self._createAgent('best_player', self.best_NN)
        
        self._loadCNN()
        if config.INITIAL_MEMORY_VERSION is None:
            self.memory = Memory(config)
        else:
            print('LOADING MEMORY VERSION ' + str(config.INITIAL_MEMORY_VERSION) + '...')
            self.memory = Memory.load(
                config.run_archive_folder + self.gameEnv.name + '/run' + str(config.INITIAL_RUN_NUMBER).zfill(4) +
                "/memory/memory" + str(config.INITIAL_MEMORY_VERSION).zfill(4) + ".p")
    
    def _loadCNN(self) -> None:
        # If loading an existing neural network, set the weights from that model
        if config.INITIAL_MODEL_VERSION is not None:
            print('LOADING MODEL VERSION ' + str(config.INITIAL_MODEL_VERSION) + '...')
            
            self.best_player_version = config.INITIAL_MODEL_VERSION
            m_tmp = Gen_Model.read(self.gameEnv.name, config.INITIAL_RUN_NUMBER, self.best_player_version)
            self.current_NN.model.set_weights(m_tmp.get_weights())
        
        self.best_NN.model.set_weights(self.current_NN.model.get_weights())
    
    def _createCNN(self) -> Residual_CNN:
        return Residual_CNN(config.REG_CONST, config.LEARNING_RATE,
                            input_dim=(2,) + self.gameEnv.grid_shape,
                            output_dim=self.gameEnv.action_size,
                            hidden_layers=config.HIDDEN_CNN_LAYERS)
    
    def _createAgent(self, name: str, CNN: Residual_CNN) -> Agent:
        return Agent(name, self.gameEnv.state_size, self.gameEnv.action_size, config.MCTS_SIMS, config.CPUCT, CNN)
    
    def playMatchesBetweenVersions(self, env, run_version, player1version, player2version, EPISODES,
                                   logger, turns_until_tau0, goes_first=0):
        if player1version == -1:
            player1 = User('player1', env.state_size, env.action_size)
        else:
            player1_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE,
                                      input_dim=env.input_shape,  # why the input shape is here? (see _createCNN)
                                      output_dim=env.action_size,
                                      hidden_layers=config.HIDDEN_CNN_LAYERS)
            
            if player1version > 0:
                player1_network = player1_NN.read(env.name, run_version, player1version)
                player1_NN.model.set_weights(player1_network.get_weights())
            player1 = Agent('player1', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, player1_NN)
        
        if player2version == -1:
            player2 = User('player2', env.state_size, env.action_size)
        else:
            player2_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, env.input_shape, env.action_size,
                                      config.HIDDEN_CNN_LAYERS)
            
            if player2version > 0:
                player2_network = player2_NN.read(env.name, run_version, player2version)
                player2_NN.model.set_weights(player2_network.get_weights())
            player2 = Agent('player2', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, player2_NN)
        
        results, memory = self.playMatches(player1, player2, EPISODES, logger, turns_until_tau0, None, goes_first)
        
        return results, memory
    
    def playMatches(self, player1: Agent, player2: Agent, EPISODES, logger, turns_until_tau0, memory: Memory = None, goes_first=0):
        gameEnv = self.gameProvider()
        results = Results(player1, player2)
        
        for e in range(EPISODES):
            
            logger.info('====================')
            logger.info('EPISODE %d OF %d', e + 1, EPISODES)
            logger.info('====================')
            
            print(str(e + 1) + ' ', end='')
            
            state = gameEnv.reset()
            
            turn = 0
            player1.mcts = None
            player2.mcts = None
            
            player1Starts = random.randint(0, 1) * 2 - 1 if goes_first == 0 else goes_first
            
            players = {
                player1Starts * 1: {"agent": player1, "name": player1.name},
                player1Starts * -1: {"agent": player2, "name": player2.name}
            }
            
            results.setStart(players[1]["agent"])
            logger.info(players[1]["name"] + ' plays as X')
            logger.info('--------------')
            
            gameEnv.gameState.render(logger)
            
            while not state.isFinish():
                turn = turn + 1
                
                # ### Run the MCTS algo and return an action
                if turn < turns_until_tau0:
                    action, pi, MCTS_value, NN_value = players[state.playerTurn]['agent'].act(state, 1)
                else:
                    action, pi, MCTS_value, NN_value = players[state.playerTurn]['agent'].act(state, 0)
                
                if memory is not None:
                    # ###Commit the move to memory
                    memory.commit_stmemory(gameEnv.identities(state, pi))
                
                logger.info('action: %d', action)
                for r in range(gameEnv.grid_shape[0]):
                    logger.info(['----' if x == 0 else '{0:.2f}'.format(np.round(x, 2)) for x in
                                 pi[gameEnv.grid_shape[1] * r: (gameEnv.grid_shape[1] * r + gameEnv.grid_shape[1])]])
                logger.info('MCTS perceived value for %s: %f', state.pieces[str(state.playerTurn)],
                            np.round(MCTS_value, 2))
                logger.info('NN perceived value for %s: %f', state.pieces[str(state.playerTurn)], np.round(NN_value, 2))
                logger.info('====================')
                
                # ## Do the action
                state, _ = gameEnv.step(action)
                # the value of the newState from the POV of the new playerTurn i.e.
                # value = -1 if the previous player played a winning move
                
                gameEnv.gameState.render(logger)
                if state.isFinish():
                    if memory is not None:
                        # # ## If the game is finished, assign the values correctly to the game moves
                        memory.commit_ltmemory(state.playerTurn, state.getValue())
                    
                    if state.getValue() == 1:
                        logger.info('%s WINS!', players[state.playerTurn]['name'])
                        results.won(players[state.playerTurn]['agent'])
                    
                    elif state.getValue() == -1:
                        logger.info('%s WINS!', players[-state.playerTurn]['name'])
                        results.won(players[-state.playerTurn]['agent'])
                    
                    else:
                        logger.info('DRAW...')
                        results.draw()
                    
                    results.addPoints(players[state.playerTurn]['agent'], state.score[0])
                    results.addPoints(players[-state.playerTurn]['agent'], state.score[1])
        
        return results, memory
    
    def iterate(self) -> None:
        iteration = 0
        while 1:
            
            iteration += 1
            
            print('ITERATION NUMBER ' + str(iteration))
            
            lg.logger_main.info('BEST PLAYER VERSION: %d', self.best_player_version)
            print('BEST PLAYER VERSION ' + str(self.best_player_version))
            
            # ####### SELF PLAY ########
            print('SELF PLAYING ' + str(config.EPISODES) + ' EPISODES...')
            _, memory = self.playMatches(self.best_player, self.best_player, config.EPISODES, lg.logger_main,
                                         turns_until_tau0=config.TURNS_UNTIL_TAU0, memory=self.memory)
            print('\n')
            
            memory.clear_stmemory()
            
            if memory.isFull():
                
                # ####### RETRAINING ########
                print('RETRAINING...')
                self.current_player.replay(memory)
                print('')
                
                if iteration % 5 == 0:
                    strIteration = str(iteration).zfill(4)
                    pickle.dump(memory, open((config.outRunPath(f"memory/memory{strIteration}.p")), "wb"))
                
                lg.logger_memory.info('====================')
                lg.logger_memory.info('NEW MEMORIES')
                lg.logger_memory.info('====================')
                
                memory_samp = memory.sample()
                
                for s in memory_samp:
                    current_value, current_probs, _ = self.current_player.get_preds(s['state'])
                    best_value, best_probs, _ = self.best_player.get_preds(s['state'])
                    
                    lg.logger_memory.info('MCTS VALUE FOR %s: %f', s['playerTurn'], s['value'])
                    lg.logger_memory.info('CUR PRED VALUE FOR %s: %f', s['playerTurn'], current_value)
                    lg.logger_memory.info('BES PRED VALUE FOR %s: %f', s['playerTurn'], best_value)
                    lg.logger_memory.info('THE MCTS ACTION VALUES: %s', ['%.2f' % elem for elem in s['AV']])
                    lg.logger_memory.info('CUR PRED ACTION VALUES: %s', ['%.2f' % elem for elem in current_probs])
                    lg.logger_memory.info('BES PRED ACTION VALUES: %s', ['%.2f' % elem for elem in best_probs])
                    lg.logger_memory.info('ID: %s', s['state'].id)
                    lg.logger_memory.info('INPUT TO MODEL: %s',
                                          self.current_player.model.convertToModelInput(s['state']))
                    
                    s['state'].render(lg.logger_memory)
                
                # ####### TOURNAMENT ########
                print('TOURNAMENT...')
                results, _ = self.playMatches(self.best_player, self.current_player, config.EVAL_EPISODES,
                                              lg.logger_tourney,
                                              turns_until_tau0=0, memory=None)
                print('\nSCORES')
                print(results)
                
                print('\n\n')
                if results.scores(self.current_player) > results.scores(self.current_player) * config.SCORING_THRESHOLD:
                    self.best_player_version = self.best_player_version + 1
                    self.best_NN.model.set_weights(self.current_NN.model.get_weights())
                    self.best_NN.write(self.gameEnv.name, self.best_player_version)
            
            else:
                print('MEMORY SIZE: ' + str(len(memory.long_term_memory)))


class Results:
    def __init__(self, player1: Agent, player2: Agent):
        self.players = {
            player1: {'won': 0, 'start': 0, 'points': []},
            player2: {'won': 0, 'start': 0, 'points': []},
        }
        self.drawn = 0
        self.start = None
    
    def setStart(self, player: Agent):
        self.start = player
    
    def won(self, player: Agent):
        self.players[player]['won'] = self.players[player]['won'] + 1
        if player == self.start:
            self.players[player]['start'] = self.players[player]['start'] + 1
    
    def draw(self):
        self.drawn = self.drawn + 1
    
    def addPoints(self, player: Agent, points):
        self.players[player]['points'].append(points)
    
    def scores(self, player: Agent):
        return self.players[player]['won']
    
    def __str__(self):
        return str(self.players)  # TODO improve this
