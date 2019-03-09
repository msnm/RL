from abc import ABC, abstractmethod
import Percept


class LearningStrategy(ABC):

    def learn(self, percept: Percept):
        self.evaluate(percept)
        self.improve()

    @abstractmethod
    def evaluate(self, percept: Percept):
        return percept

    @abstractmethod
    def next_action(self):
        print(self)

    def improve(self):
        pass