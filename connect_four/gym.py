from .player import RandomAgent, SmartAgent_2, SimpleAgent
from .game import Game
from .models import ValueNet
import os
import sys
import numpy as np
from .constants import *
from tensorflow import convert_to_tensor
import random

GAMMA = .75

def field_to_categorical_2(a):
    return np.stack([(a == -1).astype(float), (a == 1).astype(float)], 2)

def train(epochs:int):
    model = ValueNet()

    start_mse = 0
    end_mse = 0
    G = []
    S = []
    for i in range(epochs):
        #epsilon = i*(1-.9)/epochs + .92
        #epsilon = i/epochs
        agent_1 = SmartAgent_2(model, .7)
        agent_2 = SmartAgent_2(model, .7)
        game = Game(agent_1, agent_2)
        game.simulate()

        # if g.is_random[-1]:
        #     continue

        g_, s_ = get_learning_data(game)
        G.extend(g_)
        S.extend(s_)

        if (i+1) % 500 == 0:
            start_mse, end_mse = learn(S, G, model)
            G = []
            S = []

        sys.stdout.write("\repochs: %i \t start mse: %.2f \t end mse: %.2f" % ((i+1), start_mse, end_mse))
        #sys.stdout.write("\n\rmoves: %s" % (moves/moves.sum()))
        sys.stdout.flush()

        if (i+1) % 500 == 0:
            model.save_weights(os.path.join('data', 'weights.h5'))
    model.save_weights(os.path.join('data', 'weights.h5'))
    #np.save(os.path.join('data', 'mse'), np.array(mses))

def benchmark(epochs:int):
    model = ValueNet()
    bm = ValueNet()

    won = 0
    draw = 0
    for i in range(epochs):
        bm_agent = SmartAgent_2(bm, weights_name='weights_benchmark.h5')
        smart_agent = SmartAgent_2(model, weights_name='weights.h5')

        # ensure the position doesn't play a role
        if random.random() < .5:
            g = Game(bm_agent, smart_agent)
            result = g.simulate()
            if result == 1:
                won += 1
            elif result == 0:
                draw += 1
        else:
            g = Game(smart_agent, bm_agent)
            result = g.simulate()
            if result == -1:
                won += 1
            elif result == 0:
                draw += 1
        sys.stdout.write("\repochs: {:d} \t won: {:.2f} \t drawn: {:.2f}".format((i+1), won / (i+1), draw / (i+1)))
        sys.stdout.flush()

def get_learning_data(g:Game):
    rewards = g.rewards 
    states = g.states
    is_random = g.is_random

    # dispose moves before last random moves
    rewards = rewards[len(is_random)-1-np.argmax(is_random[::-1]):]
    states = states[len(is_random)-1-np.argmax(is_random[::-1]):]

    states.extend([-s for s in states])

    batch_size = len(states)
    rewards.reverse()
    G = []
    sum_reward = 0
    for r in rewards:
        sum_reward = r + GAMMA*sum_reward
        G.append(sum_reward)
    G.reverse()
    G.extend([-g for g in G])
    S = [field_to_categorical_2(s) for s in states]
    return G, S

def learn(S, G, value_net):
    batch_size = len(G)
    G = convert_to_tensor(G)
    S = convert_to_tensor(S)

    res = value_net.fit(S, G, batch_size=batch_size, epochs=100, verbose=0)
        
    return res.history['loss'][0], res.history['loss'][-1]

# https://towardsdatascience.com/reinforce-policy-gradient-with-tensorflow2-x-be1dea695f24