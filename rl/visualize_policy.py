from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.app import App
from kivy.clock import Clock
from rl.agent import Agent
import numpy as np
import random


class FrozenLakeWidget(GridLayout):

    arrows = ['&larr;', '&darr;', '&rarr;', '&uarr;']

    def __init__(self, agent: Agent, **kwargs):
        super(FrozenLakeWidget, self).__init__(**kwargs)
        self.agent = agent
        policy = self._visualpolicy()
        self.cols = 4
        for action in policy:
            self.add_widget(
                Button(text=str(action),
                      font_size='40sp',
                      font_name='/Library/Fonts/Arial Black.ttf',
                       background_color=[random.random(),random.random(),random.random(),random.random()])) #Font is system specific!

    def update(self):
        policy = self._visualpolicy()
        self.clear_widgets()
        for index, action in enumerate(policy):
            if index == self.agent.state:
                self.add_widget(
                    Button(text=str(action),
                          font_size='40sp',
                          font_name='/Library/Fonts/Arial Black.ttf',
                           background_color=[0, 0, 1, 1]))
            else:
                self.add_widget(
                    Button(text=str(action),
                           font_size='40sp',
                           font_name='/Library/Fonts/Arial Black.ttf',
                           background_color=[42/255, 206/255, 1, 1]))

    def _visualpolicy(self):
        policy_direction = np.array([action.argmax() for action in self.agent.learning_strategy.policy])
        return self._nr_to_arrow(policy_direction)

    # LEFT = 0, DOWN = 1 RIGHT = 2  UP = 3
    def _nr_to_arrow(self, policy):
        policy_str = policy.tolist()
        for i, val in enumerate(policy_str):
            if i == 5 or i == 7 or i == 11 or i == 12:
                policy_str[i] = 'H'
            elif val == 0:
                policy_str[i] = u"\u2190"
            elif val == 1:
                policy_str[i] = u"\u2193"
            elif val == 2:
                policy_str[i] = u"\u2192"
            elif val == 3:
                policy_str[i] = u"\u2191"
        policy_str[-1] = 'G'
        return policy_str




class MyApp(App):

    def __init__(self, agent: Agent):
        super().__init__()
        self.agent = agent
        self.frozen_lake = FrozenLakeWidget(agent)
        Clock.schedule_interval(self.update, 1 / 30)

    def build(self):
        return self.frozen_lake

    def update(self, dt):
        self.frozen_lake.update()
        self.title = 'Episode: {} Epsilon: {} Average Reward (last 1000): {}'.format(str(self.agent.episode), str(self.agent.learning_strategy.epsilon), str(self.agent.average_reward_over_n(1000)))
