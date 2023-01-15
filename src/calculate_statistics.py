import numpy as np
from game_objects import NeuralPlayer, RandomPlayer, Game, Match
import sys
import tensorflow as tf
from tqdm import tqdm

model_paths = sys.argv[1:]
models = [tf.keras.models.load_model(s) for s in model_paths]

players = [RandomPlayer()]

N_games = 1000

for model in models:
    players.append(NeuralPlayer(model))

win_matrix = np.zeros((len(players), len(players)))


for i, player1 in tqdm(enumerate(players)):
    for j, player2 in enumerate(players):


        player_1_wins = 0
 
        m = Match(player1, player2)

        for _ in range(N_games):

            outcome, history = m.play_game(starting_player=1)

            if outcome == 1:
                player_1_wins += 1


        win_matrix[i, j] = player_1_wins/N_games

## Rows refer to the starting player. Columns refer to the second player
np.savetxt("win_matrix.dat", win_matrix)