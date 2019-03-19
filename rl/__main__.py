
from rl.visualize_policy import MyApp
from tests.test_agent import TestAgent
from threading import Thread

if __name__ == "__main__":
    test_agent = TestAgent()
    test_agent.test_qlearning()
    MyApp(test_agent.agent).run()



