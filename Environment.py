from abc import ABC, abstractmethod
import gym


class Environment(ABC):

    def __init__(self):
        self._state = None

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def n_states(self):
        pass

    @abstractmethod
    def n_actions(self):
        pass


class FrozenLake(Environment):

    def __init__(self):
        super().__init__()
        self._env = gym.make('FrozenLake-v0')

    def reset(self):
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def n_states(self):
        return self._env.observation_space.n

    def n_actions(self):
        return self._env.action_space.n

