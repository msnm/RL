from abc import ABC, abstractmethod
import gym
from rl.percept import Percept


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
    def step(self, action, state):
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
        self._n_states = self._env.observation_space.n
        self._n_actions = self._env.action_space.n

    def reset(self):
        return self._env.reset()

    def step(self, action, state):
        percept = [state, action]
        return Percept(tuple(percept + list(self._env.step(action))))

    @property
    def env(self):
        return self._env

    @property
    def n_states(self):
        return self._n_states

    @property
    def n_actions(self):
        return self._n_actions

class Taxi(Environment):

    def __init__(self):
        super().__init__()
        self._env = gym.make('Taxi-v2')
        self._n_states = self._env.observation_space.n
        self._n_actions = self._env.action_space.n

    def reset(self):
        return self._env.reset()

    def step(self, action, state):
        percept = [state, action]
        return Percept(tuple(percept + list(self._env.step(action))))

    @property
    def env(self):
        return self._env

    @property
    def n_states(self):
        return self._n_states

    @property
    def n_actions(self):
        return self._n_actions