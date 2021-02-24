from abc import abstractmethod, ABC


class IGame(ABC):
    def __init__(self):
        self.name = None  # TODO: This are all required and will be change to methods
        self.input_shape = None
        self.grid_shape = None
        self.action_size = None
        self.gameState = None
        self.players = []
   
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
    def allowedActions(self):
        ...

    @abstractmethod
    def binary(self):
        ...

    @abstractmethod
    def id(self):
        ...
    
    @abstractmethod
    def takeAction(self, action):
        ...

    @abstractmethod
    def render(self, logger):
        ...

    @abstractmethod
    def values(self):
        ...

    @abstractmethod
    def score(self):
        ...
    
    @abstractmethod
    def isEndGame(self):
        ...

    @abstractmethod
    def getWinner(self):
        ...

    def __getstate__(self):
        state = self.__dict__
        for key in list(state):
            if callable(state[key]):
                del state[key]
        return state
