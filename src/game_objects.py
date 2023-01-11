import numpy as np


class Game:
    
    def __init__(self):
        """
        Class that manages a game of tic-tac-toe. The board is represented as a 3 x 3 matrix.
        Players play "tokens" that take the values +1 (crosses) or -1 (circles). Unoccupied squares
        in the board are represented by zeros.
        """

        self.board = np.zeros((3,3))

    def check_win(self):
        """
        Checks if the current boardstate results in a win for a given player

        Returns:
            int: 1 if the player with the "+1" token wins, -1 if the "-1" player wins, and 0 if the
            boardstate is not a win for either player.
        """
        cols = np.sum(self.board, axis = 0)
        rows = np.sum(self.board, axis = 1)
        diag1 = np.trace(self.board)
        diag2 = np.trace(self.board[::-1])
        results = np.hstack([cols, rows, diag1, diag2])

        if np.any(results == 3):
            return 1
        elif np.any(results == -3):
            return -1
        else:
            return 0

    def get_plays(self):
        """
        Function that returns the available squares for the current boardstate

        Returns:
            list: List of tuples of the form (i,j), each representing the indices of avaliable squares
        """
        
        avaliable_spaces = []

        for i, row in enumerate(self.board):

            for index in np.where(row == 0)[0]:

                avaliable_spaces.append((i, index))

        return avaliable_spaces

    def make_play(self, position, token):
        """
        Makes a play into the board

        Args:
            position (tuple): A tuple (i, j) with the indices of the squares to fill
            token (int): The value to fill the square with. Can be 1 or -1

        Raises:
            ValueError: If the indicated square is already occupied
            ValueError: If a play other than +1 or -1 is submitted
        """
        if self.board[position] != 0:
            raise ValueError("Illegal play submitted. The square is already occupied")
        
        if token not in [1, -1]:
            raise ValueError("Illegal play submitted. The only plays are +1 or -1")
        
        self.board[position] = token 


    def reset(self):
        """
        Resets the board
        """
        self.board = np.zeros((3,3))


class NeuralPlayer:

    def __init__(self, model):
        """
        A class that emulates a player in a game of Tic-Tac-Toe. It is controlled by a trained model

        Args:
            model (tensorflow.keras.model): A model capable of predicting the winning probability of a given boardstate
        """

        self.model = model

    def predict_move(self, game, token_to_play):
        """
        Given a game and a token to play, returns the predicted position for the given token.

        Args:
            game (Game): An object of the Game() class.
            token_to_play (int): Either +1 or -1. The token to be played by NeuralPlayer

        Returns:
            tuple: (i, j), the coordinates of the square to place the token into
        """

        possible_plays = game.get_plays()
        possible_final_boardstates = np.zeros((len(possible_plays), 3 ,3))

        for i, play in enumerate(possible_plays):

            dum = np.copy(game.board)
            dum[play] = token_to_play
            possible_final_boardstates[i] = dum

        ## Models are trained to be the player with the "+1" token. Since we dont want to 
        ## double train, we need to implement an exception

        if token_to_play == 1:

            # In this case, the model is already trained to find the optimal play

            probabilities = self.model.predict(possible_final_boardstates, verbose = 0)

        elif token_to_play == -1:

            # Interchanging the +1s and -1s gives a boardstate that the model can understand

            probabilities = self.model.predict(-possible_final_boardstates, verbose = 0)

        else:

            raise ValueError("Unknown token introduced")

        return possible_plays[np.argmax(probabilities)]


class RandomPlayer:

    def __init__(self):

        self.model = "This player makes random plays"

    def predict_move(self, game, token_to_play = 1):

        possible_plays = game.get_plays()
        index = np.random.choice(np.arange(len(possible_plays)))

        return possible_plays[index]

class Match:

    def __init__(self, player1, player2):
        """
        Class to handle a match between two players

        Args:
            player1 : A class describing the player that goes first. It will play with "+1"
            player2 : A class describing the player that goes second. It will play with "-1"
        """
        self.player1  = player1
        self.player2 = player2
        self.game = Game()

    def play_game(self, starting_player):

        self.game.reset()
        outcome = self.game.check_win()
        turn = starting_player
        game_history = []


        while outcome == 0:

            if self.game.get_plays() == []:
                break
            else:
                play = self.player1.predict_move(self.game, 1) if turn == 1 else self.player2.predict_move(self.game, -1)
                self.game.make_play(play, turn)
                turn *= -1
                outcome = self.game.check_win()
                game_history.append(np.copy(self.game.board))

        return outcome, game_history



