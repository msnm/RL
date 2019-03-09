
class Percept:
    def __init__(self, percept: tuple):
        self._state, self._action, self._new_state, self._reward, self._done = percept

    @property
    def state(self):
        return self._state

    @property
    def action(self):
        return self._action

    @property
    def next_state(self):
        return self._new_state

    @property
    def reward(self):
        return self._reward

    @property
    def done(self):
        return self._done

    def __repr__(self):
        return '<in {} do {} get {} -> {}>'.format(self.state, self.action, self.reward, self.next_state)

    def __hash__(self):
        return hash((self.state, self.action, self.next_state, self.reward))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.state == other.state and self.action == other.action and \
               self.reward == other.reward and self.next_state == other.next_state
