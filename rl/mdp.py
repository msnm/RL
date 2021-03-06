import numpy as np
from rl.percept import Percept


class MDP:

    def __init__(self, n_states, n_actions):
        self._n_states = n_states
        self._n_actions = n_actions
        self._reward_model = np.zeros((self.n_states, self.n_actions))
        self._state_action_freq = np.zeros((self.n_states, self.n_actions))
        self._state_action_state_freq = np.zeros((self.n_states, self.n_states, self.n_actions))
        self._transition_model = np.zeros((self.n_states, self.n_states, self.n_actions))
        self._n = 0

    @property
    def n_states(self):
        return self._n_states

    @property
    def n_actions(self):
        return self._n_actions

    @property
    def reward_model(self):
        return self._reward_model

    @property
    def state_action_freq(self):
        return self._state_action_freq
    @property
    def state_action_state_freq(self):
        return self._state_action_state_freq

    @property
    def transition_model(self):
        return self._transition_model

    @property
    def n(self):
        return self._n

    def update(self, percept: Percept):
        self._n += 1
        self._reward_model[percept.state, percept.action] = np.average(
            [self._reward_model[(percept.state, percept.action)], percept.reward], weights=[self.n - 1 / self.n, 1 / self.n])
        self._state_action_freq[percept.state, percept.action] += 1
        self._state_action_state_freq[percept.new_state, percept.state, percept.action] += 1
        self.update_transistion_model(percept)

    def update_transistion_model(self, percept: Percept):
        for t  in range(self.n_states):
            p = self._state_action_state_freq[t, percept.state, percept.action] / self.state_action_freq[percept.state, percept.action]
            self.transition_model[t, percept.state, percept.action] = p


    def __repr__(self):
        return self.state_action_freq