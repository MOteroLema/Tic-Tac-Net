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





