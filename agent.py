import game
from game import Game
import random
import numpy as np

random.seed(13)

# saves the states as keys and the column with the best result for each player as val
policy = {}



marker = 1

# the possibility to make a random move
epsilon = 1
# decay of epsilon
decay = .999

gamma = .95

for _ in range(1000):
    g = Game()
    game_over = 0
    history = []
    moves = np.arange(game.m)

    while not bool(game_over):
        r = random.random()

        player_idx = int(marker+.5)

        pf_hash = hash(g.playfield.tostring())

        available_moves = moves[g.col_available]
        if available_moves.size == 0:
            game_over = True
            continue

        if (r < epsilon) or (not pf_hash in policy):
            # make random move
            m = random.choice(available_moves)
        else:
            m = policy[pf_hash][0]
        g.play_col(m, marker)
        r = g.check()
        if r != 0:
            # someone has won the game
            policy[pf_hash] = (m, marker*r)
            game_over = True
        else:
            history.append((pf_hash, m))
        epsilon = epsilon * decay
        marker = -marker

    history = history[::-1]
    score = gamma
    for state, m in history:
        if state in policy:
            current_score = policy[state][1]
            if current_score < score:
                # better move found
                policy[state] = (m, score)
        else:
            policy[state] = (m, score)
        score = score*gamma

pass




    

