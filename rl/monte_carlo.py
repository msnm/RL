from rl.learning_strategy import LearningStrategy
from rl.environment import Environment
from rl.percept import Percept
import numpy as np


class MCLearning(LearningStrategy):

    def __init__(self, env: Environment, learning_rate=0.01, reward_discount_rate=0.95, decay_rate=0.001, epsilon=1,
                 epsilon_min=0.01, epsilon_max=1.0):
        super().__init__(env, learning_rate, reward_discount_rate, decay_rate, epsilon, epsilon_min, epsilon_max)
        self._q_table = np.zeros((self.mdp.n_states, self.mdp.n_actions))
        self._percept_n_history = []

    def evaluate(self, percept: Percept):
        self._percept_n_history.insert(0, percept)
        self.mdp.update(percept)

        if percept.done:
            for p in self._percept_n_history:
                self._update_q(p)
            self._percept_n_history.clear()

    def _update_q(self, percept):
        q = self.q_table[percept.state, percept.action]
        a = self.learning_rate
        r = percept.reward
        y = self.reward_discount_rate
        q_a_values = self.q_table[percept.new_state, :]
        max_q = np.max(q_a_values)
        q_new = q * (1 - a) + a * (r + y * max_q)
        self.q_table[percept.state, percept.action] = q_new

    def __repr__(self):
        return 'MCLearning with '.format(super().__repr__())
