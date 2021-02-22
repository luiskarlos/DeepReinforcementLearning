from abc import abstractmethod, ABC


class IGame(ABC):
    @abstractmethod
    def reset(self):
        ...
    
    @abstractmethod
    def step(self, action):
        ...
    
    @abstractmethod
    def identities(self, state, actionValues):
        ...


class IGameState(ABC):
   
    @abstractmethod
    def _allowedActions(self):
        ...

    @abstractmethod
    def _binary(self):
        ...

    @abstractmethod
    def _convertStateToId(self):
        ...

    @abstractmethod
    def _checkForEndGame(self):
        ...

    @abstractmethod
    def _getValue(self):
        ...

    @abstractmethod
    def _getScore(self):
        ...

    @abstractmethod
    def takeAction(self, action):
        ...

    @abstractmethod
    def render(self, logger):
        ...
