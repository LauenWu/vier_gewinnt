from .player import RandomAgent, SmartAgent, SimpleAgent
from .game import Game
from .models import PolicyNet
import os
import sys
import numpy as np
from .constants import *


def train(epochs:int):
    model = PolicyNet()
    model(np.ones(N*M*3))
    try:
        model.load_weights(os.path.join('data', 'weights.h5'))
    except OSError:
        pass

    moves = np.zeros(M)
    for i in range(epochs):
        epsilon = i*(1-.75)/epochs + .75
        agent_1 = SmartAgent(model, epsilon)
        agent_2 = SmartAgent(model, epsilon)
        g = Game(agent_1, agent_2)
        g.simulate()

        moves += agent_1.learn()
        moves += agent_2.learn()

        sys.stdout.write("\repochs: %i" % (i+1))
        #sys.stdout.write("\n\rmoves: %s" % (moves/moves.sum()))
        sys.stdout.flush()
    model.save_weights(os.path.join('data', 'weights.h5'))

def benchmark(epochs:int):
    model = PolicyNet()
    model(np.ones(N*M*3))
    model.load_weights(os.path.join('data', 'weights.h5'))

    won = 0
    for i in range(epochs):
        rand_agent = RandomAgent()
        smart_agent = SmartAgent(model)
        g = Game(rand_agent, smart_agent)
        result = g.simulate()
        if result == 1:
            won += 1
        sys.stdout.write("\repochs: {:d} \t win rate: {:.2f}".format((i+1), won / (i+1)))
        sys.stdout.flush()



# https://towardsdatascience.com/reinforce-policy-gradient-with-tensorflow2-x-be1dea695f24