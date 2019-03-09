import LearningStrategy
import Percept
import MDP
import numpy as np


class QLearning(LearningStrategy):

    """
        Q-learning is an algorithm with the goal to find the optimal policy
        by learning the optimal Q-values for each state-action pair.

        Params:
            q_table <- q(n_states, n_actions) (represents policy of the agent)
                       In the beginning we know nothing about the env, thus
                       the initial value for each q-a pair is 0.
                       This table will be iteratively updated and can be used by the
                       agent to determine its next move. When in exploitation mode
                       the action with the highes Q value is chosen for a given state.
                       When in exploration mode the next_step will be random.


            learning_rate <- float value between 0 and 1
                             Determines how much information we keep from the
                             previous Q

            discount_rate <- float value between 0 and 1

        Methods:


    """

    def __init__(self, mdp: MDP, learning_rate: float, discount_rate: float):
        self._mdp = mdp
        self._q_table = np.zeros((mdp.n_states, mdp.n_actions))
        self._learning_rate = learning_rate
        self._discount_rate = discount_rate

    @property
    def mdp(self):
        return self._mdp

    @property
    def q_table(self):
        return self._q_table

    @property
    def discount_rate(self):
        return self._discount_rate

    @property
    def learning_rate(self):
        return self._learning_rate

    def evaluate(self, percept: Percept):
        self.mdp.update(percept)
        q = self.q_table[percept.state, percept.action]
        a = self.learning_rate
        r = percept.reward
        y = self.discount_rate
        self.q_table[percept.state, percept.action] = \
            q * (1 - a) + a * (r + y * np.max(self.q_table[percept.new_state, :]))
