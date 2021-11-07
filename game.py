import numpy as np

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

playfield = np.zeros((n,m))

class Game:
    def __init__(self):
        self.diags_1 = [i.copy() for i in diags_1] 
        self.diags_2 = [i.copy() for i in diags_2] 
        self.playfield = playfield.copy()
        self.col_height = np.zeros(m).astype(int)
        self.col_available = np.ones(m).astype(bool)

    def play_col(self, j:int, marker:int):
        i = self.col_height[j]
        self.playfield[i, j] = marker

        if (i,j) in diag_coords_1:
            x,y = diag_coords_1[(i,j)]
            self.diags_1[x][y] = marker
        if (i,j) in diag_coords_2:
            x,y = diag_coords_2[(i,j)]
            self.diags_2[x][y] = marker
        self.col_height[j] += 1
        self.col_available &= (self.col_height < n)



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
        

    #@jit(nopython=True)
    def check_array(self, a):
        last = 0
        count = 0
        limit = x-1
        for i in range(len(a)):
            j = a[i]
            if count == limit:
                return last
            if (j != 0) and (j == last):
                count += 1
            else:
                last = j
                count = 0
        return 0


        
    




