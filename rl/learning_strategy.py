from abc import ABC, abstractmethod
from rl.mdp import MDP
from rl.environment import Environment
from rl.percept import Percept
import numpy as np


class LearningStrategy(ABC):

    def __init__(self, env: Environment, learning_rate=0.01, reward_discount_rate=0.99, decay_rate=0.001, epsilon=1.0,
                 epsilon_min=0.01, epsilon_max=1.0):
        self._learning_rate = learning_rate
        self._reward_discount_rate = reward_discount_rate
        self._decay_rate = decay_rate
        self._mdp = MDP(env.n_states, env.n_actions)
        self._policy = np.full((self.mdp.n_states, self.mdp.n_actions), 1 / self.mdp.n_actions)
        self._q_table = None
        self._v_table = None
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_max = epsilon_max

    @property
    def mdp(self):
        return self._mdp

    @property
    def policy(self):
        return self._policy

    @property
    def q_table(self):
        return self._q_table

    @property
    def v_table(self):
        return self._v_table

    @property
    def decay_rate(self):
        return self._decay_rate

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def reward_discount_rate(self):
        return self._reward_discount_rate

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon: float):
        self._epsilon = epsilon

    @property
    def epsilon_min(self):
        return self._epsilon_min

    @property
    def epsilon_max(self):
        return self._epsilon_max

    def learn(self, percept: Percept):
        self.evaluate(percept)
        self.improve(percept)

    @abstractmethod
    def evaluate(self, percept: Percept):
        return percept

    def next_action(self, state):
        return np.random.choice(self.mdp.n_actions, p=self.policy[state, :])

    def improve(self, percept: Percept):
        # 1. What is the best action given the Q table.
        action_values = self.q_table[percept.state, :]
        action_star = np.random.choice(np.flatnonzero(action_values == action_values.max()))

        # 2. Need to loop over all the actions of the given state and update the policy u
        for i in range(self.mdp.n_actions):
            if i == action_star:
                self.policy[percept.state, i] = 1 - self.epsilon + self.epsilon / abs(self.mdp.n_actions)
            else:
                self.policy[percept.state, i] = self.epsilon / abs(self.mdp.n_actions)

    def update_exploiration_rate(self, episode_n: int):
        self._epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon) * np.exp(-self.decay_rate * episode_n)

    def __repr__(self):
        return 'learning_rate= {}, reward_discount_rate={}, decay_rate={}, epsilon={}, epsilon_min={}, epsilon_max={}'.format(
            self.learning_rate, self.reward_discount_rate, self.decay_rate, self.epsilon, self.epsilon_min,
            self.epsilon_max)