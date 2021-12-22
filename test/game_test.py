

import numpy as np
from .context import connect_four


def test_array_check():
    g = connect_four.Game(connect_four.Player(), connect_four.Player())
    assert(0 == g.check_array(np.array([0,1,0,1,1,1,0])))
    assert(1 == g.check_array(np.array([0,1,1,1,1,1,0])))
    assert(1 == g.check_array(np.array([1,1,1,1,1,1,0])))
    assert(1 == g.check_array(np.array([0,1,1,1,1,0,1])))
    assert(0 == g.check_array(np.array([0,1,-1,1,1,0,1])))
    assert(0 == g.check_array(np.array([0,0,0,0,0,0,1,1])))
    assert(1 == g.check_array(np.array([0,0,0,0,1,1,1,1])))
    assert(-1 == g.check_array(np.array([0,0,0,0,-1,-1,-1,-1])))


# def test_game_win_p1(mocker):    
    
#     p1 = connect_four.SmartAgent(connect_four.PolicyNet())
#     mocker.patch.object(
#         p1,
#         'choose_action'
#     )
#     p1.choose_action.return_value = 0
    
#     p2 = connect_four.SmartAgent(connect_four.PolicyNet())
#     mocker.patch.object(
#         p2,
#         'choose_action'
#     )
#     p2.choose_action.return_value = 1
#     g = connect_four.Game(p1, p2)

#     g.simulate()

#     assert g.game_over
#     assert g.result == -1
#     assert p1.rewards[-1] == 5
#     assert p2.rewards[-1] == -5
#     assert len(p1.states) == 4
#     assert len(p1.actions) == 4
#     assert len(p1.rewards) == 4
#     assert len(p2.states) == 3
#     assert len(p2.actions) == 3
#     assert len(p2.rewards) == 3


    
    

    


    