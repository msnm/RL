from rl.q_learning import QLearning
from rl.n_step_q_learning import NQLearning
from rl.monte_carlo import MCLearning
from rl.agent import Agent
from rl.environment import FrozenLake, Taxi
from unittest import TestCase


class TestAgent(TestCase):

    def __init__(self):
        super().__init__()
        self.agent = None

    def test_qlearning(self):
        frozen_lake_v0 = FrozenLake()
        strategy = QLearning(frozen_lake_v0, 0.2, 0.99)
        self.agent = Agent(frozen_lake_v0, strategy, n_episodes=1000)
        self.agent.start()
        self.test_policy_cumm_distr_equals_one(self.agent)

    def test_nqlearning(self):
        frozen_lake_v0 = FrozenLake()
        strategy = NQLearning(8, frozen_lake_v0, 0.1, 0.99)
        self.agent = Agent(frozen_lake_v0, strategy, n_episodes=30000)
        self.agent.start()
        self.test_policy_cumm_distr_equals_one(self.agent)

    def test_mclearning(self):
        frozen_lake_v0 = FrozenLake()
        strategy = MCLearning(frozen_lake_v0, 0.01)
        self.agent = Agent(frozen_lake_v0, strategy, n_episodes=30000)
        self.agent.start()
        self.test_policy_cumm_distr_equals_one(self.agent)

    def test_policy_cumm_distr_equals_one(self, agent: Agent):
        policy = agent.learning_strategy.policy
        [self.assertAlmostEqual(sum(i), 1) for i in policy]

    def test_taxi(self):
        taxi_v2 = Taxi()
        strategy = NQLearning(8, taxi_v2, 0.1, 0.99)
        self.agent = Agent(taxi_v2, strategy, n_episodes=1000)
        self.agent.start()
        self.agent.join()
        self.test_policy_cumm_distr_equals_one(self.agent)
        print(str(self.agent.learning_strategy.mdp.state_action_freq))
        print(str(self.agent.learning_strategy.mdp.reward_model))
        print(str(self.agent.learning_strategy.mdp.transition_model))


if __name__ == "__main__":
    test_agent = TestAgent()
    test_agent.test_taxi()