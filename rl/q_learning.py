from rl.learning_strategy import LearningStrategy
from rl.environment import Environment
from rl.percept import Percept

import numpy as np


class QLearning(LearningStrategy):

    """
        Q-learning is an algorithm with the goal to find the optimal policy
        by learning the optimal Q-values for each state-action pair.

        Params:
            q_table <- q(n_states, n_actions)
                       In the beginning we know nothing about the env, thus
                       the initial value for each q-a pair is 0.
                       This table will be iteratively updated and can be used by the
                       agent to determine its next move. When in exploitation mode
                       the action with the highest Q value is chosen for a given state.
                       When in exploration mode the next_step will be random.


            learning_rate <- float value between 0 and 1
                             Determines how much information we keep from the
                             previous Q

            discount_rate <- float value between 0 and 1

        Methods:
    """

    def __init__(self, env: Environment, learning_rate=0.5, reward_discount_rate=0.99, decay_rate=0.001, epsilon=1, epsilon_min=0.01, epsilon_max=1.0):
        super().__init__(env, learning_rate, reward_discount_rate, decay_rate, epsilon, epsilon_min, epsilon_max)
        self._q_table = np.zeros((self.mdp.n_states, self.mdp.n_actions))


    def evaluate(self, percept: Percept):
        #print(percept.__repr__())
        self.mdp.update(percept)
        q = self.q_table[percept.state, percept.action]
        a = self.learning_rate
        r = percept.reward
        y = self.reward_discount_rate
        q_a_values = self.q_table[percept.new_state, :]
        max = np.max(q_a_values)
        q_new = q * (1 - a) + a * (r + y * max)
        self.q_table[percept.state, percept.action] = q_new
        #print(self.q_table)
        #print("q_old ", q, " q_new ", self.q_table[percept.state, percept.action], q_new)

    #https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Evaluation%20Solution.ipynb
    #https://harderchoices.com/2018/04/04/monte-carlo-method-in-python/