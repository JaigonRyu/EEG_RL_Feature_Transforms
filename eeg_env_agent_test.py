import itertools
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial.transform import Rotation

class Env:
    def __init__(self, features,labels, num_rotations, clf, min_features=20):
        self.features = features
        self.labels = labels

        self.num_rotations = num_rotations
        self.clf = clf

        self.min_features = min_features
        self.subsets = self.get_less_subsets(features, min_features)
        self.state_space = [(subset, rotation) for subset in self.subsets for rotation in range(num_rotations)]
        self.reset()

    def get_less_subsets(self, features, min_features):
        subsets = []
        for r in range(min_features, len(features) + 1):
            subsets += list(itertools.combinations(features, r))
        return subsets

    def apply_rotation(self, subset, rotation):
        
        ## add later


        return subset

    def reset(self):
        self.current_state_index = np.random.randint(len(self.state_space))
        return self.state_space[self.current_state_index]

    def step(self, action):
        # action: index of the new state in the state_space
        self.current_state_index = action
        new_state = self.state_space[self.current_state_index]

        ## add classifier here
        new_clf = self.clf

        new_clf.fit(self.features, self.labels)

        #split the data
        

        reward = self.calculate_accuracy(new_state)
        done = False  

        return new_state, reward, done

    def calculate_accuracy(self, state):
        subset, rotation = state


        # Placeholder for classification accuracy calculation from the clf

        accuracy = 0.0 
        return accuracy

    def get_state_space_size(self):
        return len(self.state_space)

    def get_action_space_size(self):
        return len(self.state_space)
    

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class DQNAgent:
    def __init__(self, state_space_size, action_space_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=32, memory_size=10000):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = []
        self.memory_size = memory_size

        self.model = self.build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def build_model(self):
        # Simple neural network with one hidden layer
        model = nn.Sequential(
            nn.Linear(self.state_space_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_space_size)
        )
        return model

    def store_experience(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space_size)
        state = torch.FloatTensor(state)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state)
                target += self.gamma * torch.max(self.model(next_state)).item()

            state = torch.FloatTensor(state)
            target_f = self.model(state)
            target_f[action] = target

            self.model.train()
            output = self.model(state)
            loss = self.loss_fn(output, target_f)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


features = ...
num_rotations = ...
components = ...
clf = LinearDiscriminantAnalysis(n_components=components)
min_features = 100
env = Env(features, num_rotations, clf, min_features)
agent = DQNAgent(env.get_state_space_size(), env.get_action_space_size())


num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.store_experience(state, action, reward, next_state, done)
        agent.train()

        state = next_state
        total_reward += reward

    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

print("Training completed!")