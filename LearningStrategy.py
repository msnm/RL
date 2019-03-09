from abc import ABC, abstractmethod
import Percept


class LearningStrategy(ABC):

    def learn(self, percept: Percept):
        self.evaluate(percept)
        self.learn()

    @abstractmethod
    def evaluate(self, percept: Percept):
        print(self)

    @abstractmethod
    def next_action(self):
        print(self)

    def improve(self):
        print(self)