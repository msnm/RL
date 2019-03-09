import LearningStrategy
import Environment
import Percept


class Agent:

    def __init__(self, env: Environment, learning_strategy: LearningStrategy):
        self._env = env
        self._learningStrategy = learning_strategy

    @property
    def env(self):
        return self._env

    @property
    def learning_strategy(self):
        return self._learningStrategy

    def learn(self, n_episodes):
        count_episodes = 0
        while count_episodes < n_episodes:
            state = self.env.state
            while True:
                action = self.learning_strategy.next_action()
                percept = self.env.step(action)
                self.learning_strategy.learn(percept)
                state = percept.next_state #Voorlopig geen functie, kunnen we mss gebruiken om te visualiseren hoe de agent rond wandelt
                if percept.done:
                    break
        






"""
env = gym.make('FrozenLake-v0')
print("#actions = " + str(env.action_space)) # LEFT = 0, DOWN = 1 RIGHT = 2  UP = 3
print("#states = " + str(env.observation_space))
observation = env.reset()
print("Initial state:  " + str(observation)) #Should be zero (starting position)


done = False
while done == False:
    print("##### Before action #####")
    print("Current state = " + str(observation))

    print("##### Taking a random action #####")
    action = env.action_space.sample()
    print("action = " + str(action))
    observation, reward, done, info = env.step(action)

    print("##### Random action result in new state and a reward #####")
    print("New state = " + str(observation))
    env.render()
    print("Reward = " + str(reward))
    print("Final state = " + str(done))
    print(info)


#https://github.com/aaksham/frozenlake/blob/master/example.py an example to write clean code for the above gym env

"""


