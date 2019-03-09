from abc import ABC, abstractmethod


class Environment(ABC):

    def __init__(self, init_state):
        self._state = init_state

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state

    @abstractmethod
    def step(self, action):
        print(action)
