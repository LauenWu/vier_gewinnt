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

def field_to_categorical_2(a):
    return np.stack([(a == -1).astype(float), (a == 1).astype(float)], 2)

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

        # -1: player_1 wins, 1: player_2 wins, 0: draw
        self.result = 0

        # 1 is always the moving players' perspective and -1 is always the opponent
        self.states = []
        # 1 is always the moving players' perspective and -1 is always the opponent
        self.rewards = []
        self.is_random = []

    def simulate(self) -> int:
        '''
        returns: int
            returns the sign of the winning player (-1 is player_1, 1 is player_2)
        '''
        while not self.game_over:
            self.computer_move()

        # if self.result != 0:
        #     # punish losing players' last move
        #     self.marker = -self.result
        #     loser = self.__current_player()
        #     loser.store_transition(state=self.playfield.copy(), reward=-1)

        return self.result

    def human_move(self, j:int):
        if self.game_over:
            return
        assert not self.sim

        # human player's move
        player = self.__current_player()
        assert player.human

        self.play_col(j)
        if self.game_over:
            return
        
        # computer player's move
        self.computer_move()


    def computer_move(self) -> int:
        player = self.__current_player()
        assert not player.human

        action, is_random = player.choose_action(self.playfield)
        reward = self.play_col(action)

        self.is_random.append(is_random)
        # states and rewards are always stored from the perspective of player 2
        self.states.append(self.playfield.copy())
        self.rewards.append(player.sign * reward)

    def play_col(self, j:int):
        '''returns the immediate reward from this move'''
        if not (j in self.get_available_moves()):
            print('illegal move')
            # illegal move, other player wins
            self.game_over = True
            self.result = -self.marker
            return -1

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

        if self.is_won():
            # current player won
            self.game_over = True
            self.result = self.marker
            return 1

        if not max(self.col_available):
            # draw
            self.game_over = True
            return 0
        
        self.marker = -self.marker
        return 0

    def get_available_moves(self):
        return MOVES[self.col_available]

    def is_won(self):
        '''returns True if the game is won, else returns False'''
        won = False
        for i in self.diags_1:
            if won:
                return True
            won = bool(self.check_array(i))
        for i in self.diags_2:
            if won:
                return True
            won = bool(self.check_array(i))
        for i in self.playfield:
            if won:
                return True
            won = bool(self.check_array(i))
        for i in self.playfield.T:
            if won:
                return True
            won = bool(self.check_array(i))
        return won

    def check_array(self, a):
        '''returns the marker that appears 4 times in a row'''
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



        
    




