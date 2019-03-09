import numpy as np
import Percept


class MDP:

    def __init__(self, n_states, n_actions):
        self._n_states = n_states
        self._n_actions = n_actions
        self._reward
        self._state_action_freq
        self._state_action_state_freq
        self._transition_model = np.zeros({self._n_states, self._n_actions, self._n_states})

    @property
    def n_states(self):
        return  self._n_states

    @property
    def n_actions(self):
        return  self._n_actions

    def update(self, percept: Percept):
        pass