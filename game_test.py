from game import Game
import numpy as np


def test_array_check():
    g = Game()
    assert(0 == g.check_array(np.array([0,1,0,1,1,1,0])))
    assert(1 == g.check_array(np.array([0,1,1,1,1,1,0])))
    assert(1 == g.check_array(np.array([1,1,1,1,1,1,0])))
    assert(1 == g.check_array(np.array([0,1,1,1,1,0,1])))
    assert(0 == g.check_array(np.array([0,1,-1,1,1,0,1])))
    assert(0 == g.check_array(np.array([0,0,0,0,0,0,1,1])))


def test_game():
    g = Game()
    g.play_col(2,1)
    g.play_col(2,1)
    g.play_col(2,1)
    g.play_col(2,1)
    assert(1 == g.check())

    g = Game()
    g.play_col(2,1)
    g.play_col(2,-1)
    g.play_col(2,1)
    g.play_col(2,1)
    assert(0 == g.check())

    g = Game()
    g.play_col(0,-1)
    g.play_col(1,1)
    g.play_col(2,-1)
    g.play_col(2,1)
    g.play_col(3,-1)
    g.play_col(3,-1)
    g.play_col(3,1)
    g.play_col(4,-1)
    g.play_col(4,1)
    g.play_col(4,-1)
    g.play_col(4,1)
    assert(1 == g.check())

    g = Game()
    g.play_col(6,-1)
    g.play_col(5,1)
    g.play_col(5,-1)
    g.play_col(4,1)
    g.play_col(4,1)
    g.play_col(4,-1)
    g.play_col(3,1)
    g.play_col(3,-1)
    g.play_col(3,1)
    g.play_col(3,1)

    g.play_col(6,-1)
    g.play_col(5,-1)
    g.play_col(4,-1)
    g.play_col(3,-1)
    
    assert(-1 == g.check())

    g = Game()
    g.play_col(5,-1)
    g.play_col(5,1)
    g.play_col(5,-1)
    g.play_col(5,-1)
    g.play_col(5,-1)
    g.play_col(5,-1)
    
    assert(-1 == g.check())