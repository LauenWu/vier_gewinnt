from logging import disable
import random
import numpy as np
from .models import PolicyNet, ValueNet
from .constants import *
from tensorflow.math import log, is_nan, square
from tensorflow import matmul, where, convert_to_tensor, GradientTape, float32, squeeze, expand_dims
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow_probability as tfp
import os

def field_to_categorical(a):
    b = np.zeros((N,M,3))
    for i in range(N):
        for j in range(M):
            b[i,j,int(a[i,j])+1] = 1
    return b

def field_to_categorical_2(a):
    return np.stack([(a == -1).astype(float), (a == 1).astype(float)], 2)

GAMMA = .75

random.seed(13)

class Player:
    def __init__(self):
        self.human = False
        pass

    def choose_action(self, playfield:np.array):
        pass

class HumanPlayer(Player):
    def __init__(self):
        super().__init__()
        self.human = True

    def choose_action(self, playfield:np.array):
        pass

class RandomAgent(Player):
    def __init__(self):
        super().__init__()
        self.actions = [0]
        self.rewards = [0]
        self.states = [0]


    def choose_action(self, playfield:np.array) -> int:
        # randomly chooses an available move
        available_moves = np.where(playfield==0, True, False).max(0)
        return np.random.choice(MOVES[available_moves])

    def store_transition(self, playfield, action, reward):
        pass

    def learn(self):
        pass

class SimpleAgent(Player):
    def __init__(self):
        super().__init__()

    def choose_action(self, playfield:np.array) -> int:
        # randomly chooses an available move
        available_moves = np.where(playfield==0, True, False).max(0)

        return MOVES[available_moves][-1]

    def store_transition(self, playfield, action, reward):
        pass

    def learn(self):
        pass

class SmartAgent(Player):
    def __init__(self, policy_net:PolicyNet, epsilon=1):
        super().__init__()
        self.actions = []
        self.rewards = []
        self.states = []
        self.policy_net = policy_net
        self.policy_net.compile(optimizer=Adam())
        self.policy_net(np.ones((N,M,3)))
        self.epsilon = min(1, epsilon)

        # first 2 moves are random
        self.i = 0
        try:
            self.policy_net.load_weights(os.path.join('data', 'policy', 'weights.h5'))
        except OSError:
            pass
            self.policy_net.save_weights(os.path.join('data', 'policy', 'weights.h5'))

    def choose_action(self, playfield:np.array) -> int:
        # can choose illegal action
        # ensure that each player sees himself as 1 and the opponent as -1
        available_moves = np.where(playfield==0, True, False).max(0)
        self.i += 1
        if random.random() < self.epsilon and self.i > 2:
            state = self.sign * playfield
            probs = self.policy_net(field_to_categorical(state))
            # setting probs for illegal moves to 0
            # probs = np.where(playfield==0, 1, 1e-100).max(0) * probs
            action_probs = tfp.distributions.Categorical(probs=probs)
            action = action_probs.sample()
            return action.numpy()[0]
        
        return np.random.choice(MOVES[available_moves])
        

    def store_transition(self, playfield:np.array, action:int, reward:float32):
        if self.i <= 2:
            return

        # ensure that each player sees himself as 1 and the opponent as -1
        state = self.sign * playfield
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def learn(self):
        #print('moves:', to_categorical(self.actions, num_classes=M).sum(0))

        self.rewards.reverse()
        G = []
        sum_reward = 0
        for r in self.rewards:
            sum_reward = r + GAMMA*sum_reward
            G.append(sum_reward)
        G.reverse()
        G = convert_to_tensor(G)
        actions = to_categorical(self.actions, num_classes=M)

        with GradientTape() as tape:
            loss = 0
            for g, state, action in zip(G, self.states, actions):
                probs = self.policy_net(field_to_categorical(state), training=True)
                probs = where(probs < 1e-30, 1e-30, probs)
                # print(probs.numpy().min())
                # if (probs.numpy() == 0).max():
                #     print('0', probs)
                if is_nan(probs).numpy().max():
                    print('nan', probs)
                action_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(action)
                loss += -g * squeeze(log_prob)

        grad = tape.gradient(loss, self.policy_net.trainable_variables)
        self.policy_net.optimizer.apply_gradients(zip(grad, self.policy_net.trainable_variables))
        return to_categorical(self.actions, num_classes=M).sum(0)

class SmartAgent_2(Player):
    def __init__(self, value_net:ValueNet, epsilon=1, weights_name='weights.h5', debug=False):
        super().__init__()
        self.actions = []
        self.rewards = []
        self.states = []
        self.value_net = value_net
        self.value_net.compile(optimizer=Adam(), loss=MeanSquaredError())
        self.value_net(expand_dims(np.ones((N,M,2)), 0))

        self.epsilon = min(1, epsilon)
        self.debug = debug

        # first 2 moves are random
        # self.i = 0

        try:
            self.value_net.load_weights(os.path.join('data', weights_name))
        except OSError:
            print('create new weights')
            pass
            self.value_net.save_weights(os.path.join('data', weights_name))

    def choose_action(self, playfield:np.array):

        if self.debug:
            state = self.sign * playfield
            one_hot_state = field_to_categorical_2(state)
            value = self.value_net(expand_dims(one_hot_state,0))
            print('current evaluation', value)

        # can't choose illegal action
        available_moves = np.where(playfield==0, True, False).max(0)
        if random.random() < self.epsilon:
            # ensure that each player sees himself as 1 and the opponent as -1
            state = self.sign * playfield

            one_hot_state = field_to_categorical_2(state)
            action_values = -np.ones(M)
            for action in MOVES[available_moves]:
                col = state[:,action]
                row_idx = np.argmin(col != 0)
                one_hot_state[row_idx, action, 1] = 1
                action_values[action] = self.value_net(expand_dims(one_hot_state,0))
                one_hot_state[row_idx, action, 1] = 0

            if self.debug:
                print(action_values)

            return np.argmax(action_values), False
        
        return np.random.choice(MOVES[available_moves]), True

    def learn(self):
        batch_size = len(self.rewards)
        self.rewards.reverse()
        G = []
        sum_reward = 0
        for r in self.rewards:
            sum_reward = r + GAMMA*sum_reward
            G.append(sum_reward)
        G.reverse()
        G = convert_to_tensor(G)
        states = convert_to_tensor(self.states)

        res = self.value_net.fit(states, G, batch_size=batch_size, epochs=20, verbose=0)
            
        return res.history['loss'][-1]