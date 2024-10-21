import numpy as np

class LearningAutomata:
    def __init__(self, num_actions, learning_rate=0.1):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.action_probabilities = np.ones(num_actions) / num_actions

    def select_action(self):
        return np.random.choice(self.num_actions, p=self.action_probabilities)

    def update(self, action, reward):
        if reward == 1:  # Positive reward, reinforce the chosen action
            self.action_probabilities[action] += self.learning_rate * (1 - self.action_probabilities[action])
        else:  # Negative reward, penalize the chosen action
            self.action_probabilities[action] -= self.learning_rate * self.action_probabilities[action]

        self.action_probabilities /= np.sum(self.action_probabilities)
