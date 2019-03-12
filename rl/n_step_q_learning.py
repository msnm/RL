from rl.learning_strategy import LearningStrategy
from rl.environment import Environment
from rl.percept import Percept

import numpy as np


class NQLearning(LearningStrategy):

    def __init__(self, steps: int, env: Environment, learning_rate=0.01, reward_discount_rate=0.95, decay_rate=0.001, epsilon=1, epsilon_min=0.01, epsilon_max=1.0):
        super().__init__(env, learning_rate, reward_discount_rate, decay_rate, epsilon, epsilon_min, epsilon_max)
        self._q_table = np.zeros((self.mdp.n_states, self.mdp.n_actions))
        self._percept_n_history = []
        self._steps = steps


    def evaluate(self, percept: Percept):
        self._percept_n_history.insert(0, percept) # Add in front because we want to update first the most recent Q value! Because the preceding Q values can make use of the new calculated Q value in de step before.
        self.mdp.update(percept)

        if len(self._percept_n_history) >= self._steps:
            for p in self._percept_n_history:
                self._update_q(p)
            self._percept_n_history.pop(self._steps - 1)

    def _update_q(self, percept):
        q = self.q_table[percept.state, percept.action]
        a = self.learning_rate
        r = percept.reward
        y = self.reward_discount_rate
        q_a_values = self.q_table[percept.new_state, :]
        max = np.max(q_a_values)
        q_new = q * (1 - a) + a * (r + y * max)
        self.q_table[percept.state, percept.action] = q_new

    def __repr__(self):
        return 'NQLearning with n {} step and {}'.format(self._steps, super().__repr__())