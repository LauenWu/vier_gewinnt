from game import Game
import numpy as np
import agent


def test_array_check():
    g = Game(agent.Agent())
    assert(0 == g.check_array(np.array([0,1,0,1,1,1,0])))
    assert(1 == g.check_array(np.array([0,1,1,1,1,1,0])))
    assert(1 == g.check_array(np.array([1,1,1,1,1,1,0])))
    assert(1 == g.check_array(np.array([0,1,1,1,1,0,1])))
    assert(0 == g.check_array(np.array([0,1,-1,1,1,0,1])))
    assert(0 == g.check_array(np.array([0,0,0,0,0,0,1,1])))
    assert(1 == g.check_array(np.array([0,0,0,0,1,1,1,1])))
    assert(-1 == g.check_array(np.array([0,0,0,0,-1,-1,-1,-1])))
    