# -*- coding: utf-8 -*-
# %matplotlib inline

import numpy as np
from shutil import copyfile

from foundation.director import Director

from games.Connect4Game import Connect4Game
from games.ticTacToe import TTTGame

from utils import loggers as lg, config
from utils.config import run_archive_folder

# director = Director(gameProvider=lambda: Connect4Game())
director = Director(gameProvider=lambda: TTTGame())

np.set_printoptions(suppress=True)
# copy the config file to root
copyfile('./utils/config.py', config.outRunPath('config.py'))

lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')
lg.logger_main.info('=*=*=*=*=*=.      NEW LOG      =*=*=*=*=*')
lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')

# If loading an existing neural network
if config.INITIAL_RUN_NUMBER is not None:
    copyfile(run_archive_folder + director.gameEnv.name + '/run' + str(config.INITIAL_RUN_NUMBER).zfill(4) + '/config.py',
             './config.py')

# plot_model(current_NN.model, to_file=config.outRunPath('models/model.png'), show_shapes=True)
print('\n')

# ####### CREATE THE PLAYERS ########
director.iterate()

