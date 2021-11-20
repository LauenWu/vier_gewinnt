import numpy as np
from agent import Agent
import random

x = 4
n = 6
m = 7

n_diag = n + m - 2*x + 1

comp = n-x
# maps cartesian to diags
dc_1 = -np.ones((n,m,2))
dc_2 = -np.ones((n,m,2))

diag_coords_1 = {}
diag_coords_2 = {}
for i in range(n):
    for j in range(m):
        idx_diag = j-i+comp
        if idx_diag >= 0 and idx_diag < n_diag:
            diag_coords_1[(i,j)] = (idx_diag, min(i,j))
            dc_1[i,j,0] = idx_diag
            dc_1[i,j,1] = min(i,j)
            
        
        j_ = m-j-1
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

moves = np.arange(m)
playfield = np.zeros((n,m))

class Game:
    def __init__(self, agent_1:Agent, agent_2:Agent = None):
        self.diags_1 = [i.copy() for i in diags_1] 
        self.diags_2 = [i.copy() for i in diags_2] 
        self.playfield = playfield.copy()
        self.col_height = np.zeros(m).astype(int)
        self.col_available = np.ones(m).astype(bool)
        self.marker = 1
        self.game_over = False
        self.agents = (agent_1, agent_2)

    def start(self):
        self.agents[0].move(self)
        return self.check()

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
        self.col_available &= (self.col_height < n)
        self.marker = -self.marker
        res = self.check()

        next_agent = self.agents[int(self.marker + .5)]
        if res != 0:
            self.game_over = True
        elif type(next_agent) != type(None):
            # Non-human player's move
            next_agent.move(self)
        return res

    def get_available_moves(self):
        return moves[self.col_available]

    def check(self):
        res = 0
        for i in self.diags_1:
            if bool(res):
                return res
            res = self.check_array(i)
        for i in self.diags_2:
            if bool(res):
                return res
            res = self.check_array(i)
        for i in self.playfield:
            if bool(res):
                return res
            res = self.check_array(i)
        for i in self.playfield.T:
            if bool(res):
                return res
            res = self.check_array(i)
        return res

    def check_array(self, a):
        last = 0
        count = 1
        for i in a:
            if (i != 0) and (i == last):
                count += 1
            else:
                count = 1
            last = i
            if count == x:
                return last
        return 0



        
    




