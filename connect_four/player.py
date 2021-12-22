from logging import disable
import random
import numpy as np
from .models import PolicyNet
from .constants import *
from tensorflow.math import log, is_nan
from tensorflow import matmul, where, convert_to_tensor, GradientTape, float32, squeeze, expand_dims
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import os

def field_to_categorical(a):
    b = np.zeros((N,M,3))
    for i in range(N):
        for j in range(M):
            b[i,j,int(a[i,j])+1] = 1
    return b

GAMMA = .99

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
        self.epsilon = max(1, epsilon)

        # first 2 moves are random
        self.i = 0
        # try:
        #     self.policy_net.load_weights(os.path.join('data', 'weights.h5'))
        # except OSError:
        #     pass
        #     self.policy_net.save_weights(os.path.join('data', 'weights.h5'))

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

    #-----------------------------------------      
    # def loss(self, prob, action, reward):
    #     # dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
    #     # log_prob = dist.log_prob(action)
    #     # loss = -log_prob*reward
    #     loss = -log(matmul(prob, np.expand_dims(action, 0).T) + 1e-100) * reward
    #     return loss

    # def train(self, states:np.array, actions:np.array, rewards:np.array):
    #     actions = to_categorical(actions, num_classes=M)

    #     for state, action, reward in zip(states, actions, rewards):
    #         with GradientTape() as tape:
    #             p = self(state, training=True)
    #             loss = self.loss(p, action, reward)
    #         grads = tape.gradient(loss, self.trainable_variables)
    #         self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

    # def move(self, game) -> int:
    #     available = game.get_available_moves()
    #     assert available.size > 0

    #     # ensure that each player sees himself as 1 and the opponent as -1
    #     state = self.sign * game.playfield.copy()
    #     pred = self.policy_net(state).numpy()[0]
    #     # mask legal moves
    #     try:
    #         pred = np.where(game.col_available, pred, 0)
    #         if pred.sum() > 0:
    #             chosen_move = np.random.choice(MOVES, p=pred/pred.sum())
    #         else:
    #             chosen_move = np.random.choice(game.get_available_moves())
    #         self.states.append(state)
    #         self.actions.append(chosen_move)
    #         self.immediate_rewards.append(game.play_col(chosen_move))
    #     except ValueError:
    #         pass