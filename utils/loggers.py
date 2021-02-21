import logging

from utils import config


def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    
    handler = logging.FileHandler(config.outRunPath(log_file))
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger


# ## SET all LOGGER_DISABLED to True to disable logging
# ## WARNING: the mcts log file gets big quite quickly


LOGGER_DISABLED = {
    'main': False
    , 'memory': False
    , 'tourney': False
    , 'mcts': False
    , 'model': False}

logger_mcts = setup_logger('logger_mcts', 'logs/logger_mcts.log')
logger_mcts.disabled = LOGGER_DISABLED['mcts']

logger_main = setup_logger('logger_main', 'logs/logger_main.log')
logger_main.disabled = LOGGER_DISABLED['main']

logger_tourney = setup_logger('logger_tourney', 'logs/logger_tourney.log')
logger_tourney.disabled = LOGGER_DISABLED['tourney']

logger_memory = setup_logger('logger_memory', 'logs/logger_memory.log')
logger_memory.disabled = LOGGER_DISABLED['memory']

logger_model = setup_logger('logger_model', 'logs/logger_model.log')
logger_model.disabled = LOGGER_DISABLED['model']
