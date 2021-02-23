from abc import abstractmethod, ABC, abstractproperty


class IGame(ABC):
    def __init__(self):
        self.name = None  # TODO: This are all required and will be change to methods
        self.input_shape = None
        self.grid_shape = None
        self.action_size = None
        self.gameState = None
   
    @abstractmethod
    def reset(self):
        ...
    
    @abstractmethod
    def step(self, action):
        ...
    
    @abstractmethod
    def identities(self, state, actionValues):
        """
        https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning/issues/36#issuecomment-573392639
        For a data augmentation.
        To make a lot of data, the mirror versions are used.
        """
        ...


class IGameState(ABC):
    def __init__(self):
        self.id = None  # TODO: This are all required and will be change to methods
        self.playerTurn = None

    @abstractmethod
    def takeAction(self, action):
        ...

    @abstractmethod
    def render(self, logger):
        ...

    @abstractmethod
    def getValue(self):
        ...

    @abstractmethod
    def isFinish(self):
        ...
