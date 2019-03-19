from threading import Thread

from rl.learning_strategy import LearningStrategy
from rl.environment import Environment
import numpy as np


class Agent(Thread):

    def __init__(self, env: Environment, learning_strategy: LearningStrategy, n_episodes=10000):
        super().__init__()
        self._env = env
        self._learningStrategy = learning_strategy
        self._reward_all_episodes = []
        self._n_episodes = n_episodes
        self._state = None
        self._episode = 0

    def run(self):
        self.learn()
        self.print_statistics()

    @property
    def env(self):
        return self._env

    @property
    def learning_strategy(self):
        return self._learningStrategy

    @property
    def n_episodes(self):
        return self._n_episodes

    @property
    def reward_all_episodes(self):
        return self._reward_all_episodes

    @property
    def state(self):
        return self._state

    @property
    def episode(self):
        return self._episode

    def learn(self):
        for episode in range(self.n_episodes):
            self._episode = episode
            self._state = self.env.reset()
            rewards_current_episode = 0
            done = False
            while not done:
                action = self.learning_strategy.next_action(self._state)
                percept = self.env.step(action, self._state)
                self.learning_strategy.learn(percept)
                self._state = percept.new_state
                rewards_current_episode += percept.reward
                done = percept.done

            self.learning_strategy.update_exploiration_rate(episode)
            self.reward_all_episodes.append(rewards_current_episode)

    def average_reward_over_n(self, n):
        l = len(self._reward_all_episodes)
        if l >= n:
            return sum(self._reward_all_episodes[l-1-n: ]) / n
        else:
            return sum(self._reward_all_episodes) / l

    def print_statistics(self):
        rewards_per_thousand_episodes = np.split(np.array(self.reward_all_episodes), self.n_episodes / 1000)
        rewards_of_last_thousand_ep_per_hundred = np.split(rewards_per_thousand_episodes[-1], 10)

        print("+--- Average reward per 1000 episodes ---+")
        count = 1000
        for r in rewards_per_thousand_episodes:
            print(count, ":", str(sum(r) / 1000), "%")
            count += 1000

        print("+--- Last 1000 episodes average reward per 100 episodes ---+")
        count = self.n_episodes - 900
        for r in rewards_of_last_thousand_ep_per_hundred:
            print(count, ":", str(sum(r) / 100), "%")
            count += 100

        print("+--- Final Q-Table ---+")
        print(self.learning_strategy.mdp.state_action_freq)

        print("+--- Final Policy ---+")
        print(self.learning_strategy.policy)