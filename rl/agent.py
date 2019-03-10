from rl import learning_strategy, environment


class Agent:

    def __init__(self, env: environment, learning_strategy: learning_strategy, n_episodes=10000):
        self._env = env
        self._learningStrategy = learning_strategy
        self._reward_all_episodes = []
        self._n_episodes = n_episodes

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

    def learn(self):
        for episode in range(self.n_episodes):
            state = self.env.reset()
            rewards_current_episode = 0
            while True:
                action = self.learning_strategy.next_action(state)
                percept = self.env.step(action, state)
                self.learning_strategy.learn(percept)
                state = percept.new_state
                rewards_current_episode += percept.reward
                if percept.done:
                    break

            self.learning_strategy.update_exploiration_rate(episode)

            if episode!=0 and episode%1000==0:
                print("Average Reward after ", episode, " episodes: ", sum(self.reward_all_episodes[episode-1001:episode])/1000)
                print("########### Q_Table ###########")
                print(self.learning_strategy.q_table)
                print("########### Policy ########### ")
                print(self.learning_strategy.policy)
            self.reward_all_episodes.append(rewards_current_episode)

#https://github.com/aaksham/frozenlake/blob/master/example.py an example to write clean code for the above gym env


