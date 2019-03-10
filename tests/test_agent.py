from rl.q_learning import QLearning
from rl.agent import Agent
from rl.environment import FrozenLake
import numpy as np


def test_qlearning():
    frozen_lake_v0 = FrozenLake()
    strategy = QLearning(frozen_lake_v0, 0.01)
    agent = Agent(frozen_lake_v0, strategy, 100000)
    agent.learn()

    #Calculate and print the average reward per thousand episodes
    rewards_per_thousand_episodes = np.split(np.array(agent.reward_all_episodes), agent.n_episodes / 1000)

    print("Average reward per 1000 episodes")
    count = 1000
    for r in rewards_per_thousand_episodes:
        print(count, ":", str(sum(r/1000)))
        count += 1000

    print("Q-table")
    print(agent.learning_strategy.mdp.state_action_freq)


if __name__ == "__main__":
    test_qlearning()