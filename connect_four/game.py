import numpy as np
from .player import Player
from .constants import *

n_diag = N + M - 2*X + 1

comp = N-X
# maps cartesian to diags
dc_1 = -np.ones((N,M,2))
dc_2 = -np.ones((N,M,2))

diag_coords_1 = {}
diag_coords_2 = {}
for i in range(N):
    for j in range(M):
        idx_diag = j-i+comp
        if idx_diag >= 0 and idx_diag < n_diag:
            diag_coords_1[(i,j)] = (idx_diag, min(i,j))
            dc_1[i,j,0] = idx_diag
            dc_1[i,j,1] = min(i,j)
            
        
        j_ = M-j-1
        idx_diag = j_-i+comp
        if idx_diag >= 0 and idx_diag < n_diag:
            diag_coords_2[(i,j)] = (idx_diag, min(i,j_))
            dc_2[i,j,0] = idx_diag
            dc_2[i,j,1] = min(i,j_)

diags_dim_1 = np.zeros(n_diag).astype(int)
diags_dim_2 = np.zeros(n_diag).astype(int)

for i in diag_coords_1:
    idx_diag, _ = diag_coords_1[i]
    diags_dim_1[idx_diag] += 1

for i in diag_coords_2:
    idx_diag, _ = diag_coords_2[i]
    diags_dim_2[idx_diag] += 1

# list of arrays containing the numbers and to calculate score for
diags_1 = [np.zeros(i) for i in diags_dim_1]
diags_2 = [np.zeros(i) for i in diags_dim_2]

playfield = np.zeros((N,M))

class Game():
    def __init__(self, agent_1:Player, agent_2:Player):
        agent_1.sign = -1
        agent_2.sign = 1
        if agent_1.human + agent_2.human == 2:
            self.sim = True
        else:
            self.sim = False         

        # first player begins
        self.marker = -1

        self.diags_1 = [i.copy() for i in diags_1] 
        self.diags_2 = [i.copy() for i in diags_2] 
        self.playfield = playfield.copy()
        self.col_height = np.zeros(M).astype(int)
        self.col_available = np.ones(M).astype(bool)
        self.game_over = False

        # used to toggle quickly btw agents
        self.agents = (agent_1, agent_2)

    def simulate(self) -> int:
        while not self.game_over:
            res = self.computer_move()

        if abs(res) == 1:
            # a player has won, punish the other player
            player = self.__current_player()
            player.rewards[-1] = -5
        return res

    def human_move(self, j:int):
        if self.game_over:
            return
        assert not self.sim

        # human player's move
        player = self.__current_player()
        assert player.human

        res = self.play_col(j)
        if self.game_over:
            return res
        
        # computer player's move
        return self.computer_move()


    def computer_move(self) -> int:
        player = self.__current_player()
        assert not player.human
        state = self.playfield.copy()
        action = player.choose_action(self.playfield)
        if action in self.get_available_moves():
            res = self.play_col(action)
        else:
            # illegal move
            res = -5
            self.game_over = True

        player.store_transition(state, action, res)
        return res


    def play_col(self, j:int) -> int:
        i = self.col_height[j]
        self.playfield[i, j] = self.marker

        if (i,j) in diag_coords_1:
            x,y = diag_coords_1[(i,j)]
            self.diags_1[x][y] = self.marker
        if (i,j) in diag_coords_2:
            x,y = diag_coords_2[(i,j)]
            self.diags_2[x][y] = self.marker
        self.col_height[j] += 1
        self.col_available &= (self.col_height < N)
        self.marker = -self.marker
        res = self.check()

        if res == 1:
            # current player won
            self.game_over = True
            return res

        if not max(self.col_available):
            # draw
            self.game_over = True
            return .5
        return res

    def get_available_moves(self):
        return MOVES[self.col_available]

    def check(self):
        '''returns 1 if the game is won, else returns 0'''
        res = 0
        for i in self.diags_1:
            if bool(res):
                return 1
            res = self.check_array(i)
        for i in self.diags_2:
            if bool(res):
                return 1
            res = self.check_array(i)
        for i in self.playfield:
            if bool(res):
                return 1
            res = self.check_array(i)
        for i in self.playfield.T:
            if bool(res):
                return 1
            res = self.check_array(i)
        return int(res != 0)

    def check_array(self, a):
        last = 0
        count = 1
        for i in a:
            if (i != 0) and (i == last):
                count += 1
            else:
                count = 1
            last = i
            if count == X:
                return last
        return 0

    def __current_player(self):
        return self.agents[int(self.marker + .5)]



        
    




