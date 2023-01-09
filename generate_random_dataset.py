from game import Game
import numpy as np

## This program produces a starting dataset for training
## We play two random players against one another
## In each player's turn, they choose a random move between all available ones
## Once the game ends, all boards are tagged according to who won
## This approach ensures no illegal moves are made, but it does not teach the NN to not make them
## This could be achieved later by trying to make a model in which board --> board, instead of board --> play


game = Game()

## We will position the "interesting player" as the one with the "+1" tokens. 
## This will be the "player". The "-1" tokens are the "opponent"
## We train the model to play with those tokens
## This is an arbitrary choice

n_games = 10000

## There will be a total of 2 * n_games played
## In half of them the player will have the first move


## Games where the player starts


going_first_configs = []
going_first_results = []


for _ in range(n_games):
    game.reset()
    turn = 1
    output = game.check_win()
    game_configs = []
    while output == 0:

        plays = game.get_plays()
        if plays == []:
            break
        choice = np.random.choice(range(len(plays)))
        game.make_play(position=plays[choice], token=turn)
        output = game.check_win()
        game_configs.append(game.board.flatten())
        turn *= -1
    
    for config in game_configs:
        going_first_configs.append(config)
        going_first_results.append(output)

## Games where the opponent starts

going_second_configs = []
going_second_results = []


for _ in range(n_games):
    game.reset()
    turn = -1
    output = game.check_win()
    game_configs = []
    while output == 0:

        plays = game.get_plays()
        if plays == []:
            break
        choice = np.random.choice(range(len(plays)))
        game.make_play(position=plays[choice], token=turn)
        output = game.check_win()
        game_configs.append(game.board.flatten())
        turn *= -1

    for config in game_configs:
        going_second_configs.append(config)
        going_second_results.append(output)


### We now save the data

first_c = np.array(going_first_configs)
second_c = np.array(going_second_configs)

configs = np.vstack((first_c, second_c))

first_r = np.array(going_first_results)
second_r = np.array(going_second_results)

results = np.hstack((first_r, second_r))

np.savetxt("configurations.dat", configs)
np.savetxt("results.dat", results)