# ### SELF PLAY
from pathlib import Path

EPISODES = 3  # 30
MCTS_SIMS = 5  # 50
MEMORY_SIZE = 300  # 30000
MEMORY_SAMPLE_SIZE = 100

TURNS_UNTIL_TAU0 = 10  # turn on which it starts playing deterministically
CPUCT = 1  # exploration vs exploitation see https://arxiv.org/pdf/1910.13012.pdf
EPSILON = 0.2
ALPHA = 0.8

# ### RETRAINING
BATCH_SIZE = 32  # 256
EPOCHS = 1
REG_CONST = 0.0001
LEARNING_RATE = 0.1
MOMENTUM = 0.9
TRAINING_LOOPS = 10

HIDDEN_CNN_LAYERS = [
    {'filters': 75, 'kernel_size': (4, 4)},
    {'filters': 75, 'kernel_size': (4, 4)},
    {'filters': 75, 'kernel_size': (4, 4)},
    {'filters': 75, 'kernel_size': (4, 4)},
    {'filters': 75, 'kernel_size': (4, 4)},
    {'filters': 75, 'kernel_size': (4, 4)}
]

# ### EVALUATION
EVAL_EPISODES = 20
SCORING_THRESHOLD = 1.3

run_folder = './run/'
run_archive_folder = './run_archive/'


INITIAL_RUN_NUMBER = None
INITIAL_MODEL_VERSION = None
INITIAL_MEMORY_VERSION = None


def outRunPath(posfix_path):
    path = Path(run_folder, posfix_path)
    Path(path.parent).mkdir(parents=True, exist_ok=True)
    return str(path)
