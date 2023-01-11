import numpy as np
from game_objects import Game, NeuralPlayer, RandomPlayer, Match
import tensorflow as tf
from tqdm import tqdm
import os

## Specify and load the type of players

player_inputs = [ input("\nPlayer 1 (random/model): "), input("\nPlayer 2 (random/model): ") ]
n_games = int(input("\nNumber of games to be played: "))

players = []

for player in player_inputs:

    if player == "random":
        players.append(RandomPlayer())
    
    else:
        try:

            keras_model = tf.keras.models.load_model(player)
            players.append(NeuralPlayer(keras_model))
        
        except:
            
            raise ValueError(f"Input <<{player}>> not understood")

## A match is created between the virtual players

player1, player2 = players
virtual_match = Match(player1, player2)

p1_wins = 0
p2_wins = 0
draws = 0

print("\nGames are being played, please do not disturb\n")

dataset_configs = []
dataset_results = []
for _ in tqdm(range(n_games)):

    starting_player = np.random.choice([-1, 1])
    outcome, game_history = virtual_match.play_game(starting_player=starting_player)

    
    ## We save the moves of each player to train the network
    ## Important to remember that the NNs are trained to play with the +1 tokens
    ## This way, when using moves from the player with the -1 to train, we must change the sign of the board and outcome

    moves_player_1 = game_history[::2]
    moves_player_2 = game_history[1::2]

    for mp1 in moves_player_1:

        dataset_configs.append(mp1 * starting_player)
        dataset_results.append(outcome * starting_player)

    for mp2 in moves_player_2:

        dataset_configs.append(-mp2 * starting_player)
        dataset_results.append(-outcome * starting_player)


    p1_wins += int((outcome + 1)/2)
    p2_wins -= int((outcome - 1)/2)
    draws += 1 if outcome == 0 else 0

dataset_results = np.array(dataset_results)
dset_configs = np.zeros((len(dataset_configs), 3, 3))
for i, c in enumerate(dataset_configs):
    dset_configs[i] = c

### Now some statistics are printed

print(" ")
print("*"*20)
print(" ")

print(f"Player 1 won {p1_wins * 100 /n_games} % of the games\n")
print(f"Player 2 won {p2_wins * 100 /n_games} % of the games\n")
print(f"{100 * draws / n_games} % of games ended in a draw\n")

## User is prompted to save the session as a training set

save = input("\nSave training set? (y/n): ")
if save == "y":
    np.save("results.npy", dataset_results)
    np.save("configs.npy", dset_configs)



