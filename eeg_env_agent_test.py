import gym 
import numpy as np
from gym import spaces
from collections import deque
import random
from scipy.stats import special_ortho_group
from sklearn.metrics import accuracy_score


class EEGFeatureEnv(gym.Env):
    def __init__(self, X_train, y_train, other_subjects_X, other_subjects_y, classifier, max_steps_per_feature=5, decay_factor=0.99, buffer_size=50):

        super(EEGFeatureEnv, self).__init__()

        self.X_train = X_train
        self.y_train = y_train
        self.classifier = classifier

        self.tot_X = other_subjects_X
        self.tot_y = other_subjects_y

        self.curr_ind = 0
        self.feature_step = 0
        self.episode_count = 1
        self.max_steps_per_feature = max_steps_per_feature
        self.decay_factor = decay_factor

        self.best_features = {}

        self.feature_replay_buffer = deque(maxlen=buffer_size)

        self.observation_space = gym.spaces.Box(low=-100000, high=100000, shape=(X_train.shape[1],), dtype=np.float32)

        #self.action_space = gym.spaces.Discrete(10)

        #continous action space for more learning first var in (2, ) for jitter and 
        self.action_space = gym.spaces.Box(low=-550.0, high=550.0, shape=(2,), dtype=np.float32)

    def step(self, action):

        #print(f"Agent took action: {action}")

        print('beginning step')

        rotation_value, jitter_value = action

        scaling_factor = self.decay_factor ** self.episode_count
        rotation_value *= scaling_factor
        jitter_value *= scaling_factor

        feature_vector = self.transformed_X[self.curr_ind].copy()

        #feature_vector = self.apply_jitter(feature_vector, action)
        #feature_vector = self.apply_jitter(feature_vector, action)

        feature_vector = self.apply_rotation_cont(feature_vector, rotation_value)
        #feature_vector = self.apply_jitter_cont(feature_vector, jitter_value)
        
       # transformed_X = self.X_train.copy()

        self.transformed_X[self.curr_ind] = feature_vector
        self.feature_replay_buffer.append(feature_vector.copy())

        accuracy = self.compute_tot_accuracy(self.transformed_X, self.y_train)


        #print(accuracy)

        #reward = (accuracy - self.initial_acc) * 100 - 0.1 * self.compute_kl_divergence(transformed_X) - 0.1 * self.compute_subject_variability(transformed_X)

        reward = accuracy - self.initial_acc * 100 

        if reward > 0:
            reward += 250
        
        else:
            reward -= 50 
        #print(reward)

        if self.curr_ind not in self.best_features or reward > self.best_features[self.curr_ind][1]:
            self.best_features[self.curr_ind] = (feature_vector.copy(), reward)

        #print(f"Step {self.curr_ind}: Accuracy = {accuracy}, Reward = {reward}") 

        self.feature_step += 1
        if self.feature_step >= self.max_steps_per_feature:
            self.feature_step = 0
            self.curr_ind += 1
        
        done = self.curr_ind >= len(self.X_train)

        return feature_vector, reward, done, {}

    def compute_subject_variability(self, transformed_X):
        predictions = [self.classifier.predict(x) for x in self.tot_X]
        std_dev = np.std(predictions, axis=0)  # Variability across subjects
        return np.mean(std_dev)  # Lower is better

    
    from scipy.stats import special_ortho_group

    def apply_rotation(self, feature_vector, action):
        """ Apply small incremental rotations using a random orthogonal matrix. """

        rotation_levels = np.linspace(-10, 10, num=10)  # Rotation angles from -10° to 10°
        rotation_angle = rotation_levels[action]
        
        dim = len(feature_vector)  # Get the feature vector size
    
        if dim < 2:
            return feature_vector  # Can't rotate a single value
        
        # Generate a random rotation matrix for the entire feature space
        dim = len(feature_vector)
        rotation_matrix = special_ortho_group.rvs(dim)
        
        feature_vector = np.dot(rotation_matrix, feature_vector)

        return feature_vector
    
    def apply_rotation_cont(sefl, feature_vector, rotation_value):

        dim = len(feature_vector)
        if dim < 2:
            return feature_vector
        
        rotation_matrix = special_ortho_group.rvs(dim)

        scaled_rotation = np.deg2rad(rotation_value * 10)

        rotated_vector = np.dot(rotation_matrix, feature_vector) * np.cos(scaled_rotation)

        return rotated_vector

    def apply_jitter(self, feature_vector, action):
        
        levels = [0.0,0.8,0.5, 0.1,0.08, 0.05, 0.01,0.008, 0.005, 0.001]
        noise_level = levels[action]

        feature_vector += np.random.normal(0, noise_level, feature_vector.shape)
        return  feature_vector

    def apply_jitter_cont(self, feature_vector, jitter_value):

        noise_level = max(0, jitter_value * 0.1)

        feature_vector += np.random.normal(0, noise_level, feature_vector.shape)

        return feature_vector

    def compute_tot_accuracy(self, transformed_X, y_train):

        self.classifier.fit(transformed_X, y_train)
        print('not working')

        tot_acc = []

        

        for x, label in zip(self.tot_X, self.tot_y):


            y_pred = self.classifier.predict(x)

            accuracy = accuracy_score(y_pred, label)

            tot_acc.append(accuracy)

        
        mean_acc = np.mean(tot_acc)

        #print(mean_acc)

        return mean_acc

    


    def reset(self):

        print('reset started')
        #print(self.initial_acc)
        if not hasattr(self, "transformed_X"):  # Only initialize once
            self.transformed_X = self.X_train.copy()

        self.curr_ind = 0
        self.feature_step = 0
        self.episode_count +=1

        if len(self.feature_replay_buffer) > 10 and random.random() < 0.3:
            idx = random.randint(0, len(self.feature_replay_buffer) -1)
            self.transformed_X[self.curr_ind] = self.feature_replay_buffer[idx].copy()


        self.initial_acc = self.compute_tot_accuracy(self.transformed_X, self.y_train)
        #print(self.initial_acc)

        print('reset worked')

        return self.transformed_X[self.curr_ind]
    

