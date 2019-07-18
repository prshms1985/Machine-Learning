#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 10 2018

@author: Prashamsh Takkalapally

Machine Problem 2
Gen-Tic-Tac-Toe Minimax Search with alpha/beta pruning
"""

import numpy as np
import random
import math

##################################################
##################################################

# Minimax function
def  alpha_beta(
        state                           , #Current game state 
        depth        = 10               , #Maximum depth, higher values result in better choices, but longer decision times.
        alpha        = -float('inf')    , #Alpha value
        beta         = float('inf')     , #Beta value
        max_player   = True             ):#Wether this is the maximizing palyer
    
    if state.is_terminal() or depth == 0: # returns the value of state once either computer, player wins or a draw
        return state
    if max_player :# tries to maximize the heuristic function
        max_val    = - float('inf')
        new_state   = None
        for child in state.get_children():
            min_val = alpha_beta(child, depth-2, alpha, beta, False)
            if max_val < min_val.value :
                new_state = child
                max_val  = min_val.value
            alpha = max(alpha, min_val.value)
            if beta <= alpha :
                break
        return new_state
    else : # min player tries to minimize the heuristic function using minimax algorithm
        min_val    = float('inf')
        new_state   = None
        for child in state.get_children():
            max_val = alpha_beta(child, depth-2, alpha, beta, True)
            if min_val > max_val.value :
                new_state   = child
                min_val    = max_val.value
            beta = min(beta, max_val.value)
            if beta <= alpha :
                break
        return new_state

class State:
    def __init__(self, marks, move):
        self.move       = move
        self.marks      = marks
        self.children   = []
        self.value      = self.set_value()     
  
    # Funciton to get all the children states from the current state
    def get_children(self):

        if not self.children:
            s = 'O' if ((self.marks == 'O').sum() < (self.marks == 'X').sum()) \
                    else 'X'
            indices = np.where(self.marks == ' ')
            for i in range(len(indices[0])):
                _marks      = self.marks.copy()
                move        = indices[0][i], indices[1][i]
                _marks[move]= s
                self.children.append(State(_marks, move))
        return self.children

    #The heuristic value of a node.
    def set_value(self):

        lines = self.get_lines()
        p1 = 0
        p2 = 0
        for line in lines :
            if (line=='X'):
                return -1
            if (line=='O'):
                return
           '''
            cpt = (line == 'X' ).sum()
            p1 = 4**cpt if cpt else 0

            cpt = (line == 'O' ).sum()
            p2 += 2**cpt if cpt else 0
            '''
        
        return p2 - p1
    
    # Gets all winning lines
    
    def get_lines(self):
        return np.concatenate((
            self.marks , 
            self.marks.transpose(), 
            [self.marks.diagonal(), np.rot90(self.marks).diagonal()]))
    
    # Check if the s(X pr O) is in a win state
    def is_win_state(self, s):
        for line in self.get_lines():
            if (line == s).all():
                return True
        return False
    
    # Function to check if X or O wins or if no more moves are left
    def is_terminal(self):
        return  (self.marks!=' ').all() or \
                self.is_win_state('O')  or \
                self.is_win_state('X')

    
    def best(self):
        return alpha_beta(self)


##################################################
##################################################
# self class is responsible for representing the game board
class GenGameBoard: 
    
    # Constructor method - initializes each position variable and the board size
    def __init__(self, boardSize):
        self.boardSize = boardSize  # Holds the size of the board
        self.marks = np.empty((boardSize, boardSize),dtype='str')  # Holds the mark for each position
        self.marks[:,:] = ' '
    
    # Prints the game board using current marks
    def printBoard(self): 
        # Prthe column numbers
        print(' ',end='')
        for j in range(self.boardSize):
            print(" "+str(j+1), end='')
        
        
        # Prthe rows with marks
        print("")
        for i in range(self.boardSize):
            # Prthe line separating the row
            print(" ",end='')
            for j in range(self.boardSize):
                print("--",end='')
            
            print("-")

            # Prthe row number
            print(i+1,end='')
            
            # Prthe marks on self row
            for j in range(self.boardSize):
                print("|"+self.marks[i][j],end='')
            
            print("|")
                
        
        # Prthe line separating the last row
        print(" ",end='')
        for j in range(self.boardSize):
            print("--",end='')
        
        print("-")
    
    
    # Attempts to make a move given the row,col and mark
    # If move cannot be made, returns False and prints a message if mark is 'X'
    # Otherwise, returns True
    def makeMove(self, row, col, mark):
        possible = False  # Variable to hold the return value
        if row==-1 and col==-1:
            return False
        
        # Change the row,col entries to array indexes
        row = row - 1
        col = col - 1
        
        if row<0 or row>=self.boardSize or col<0 or col>=self.boardSize:
            print("Not a valid row or column!")
            return False
        
        # Check row and col, and make sure space is empty
        # If empty, set the position to the mark and change possible to True
        if self.marks[row][col] == ' ':
            self.marks[row][col] = mark
            possible = True    
        
        # Prout the message to the player if the move was not possible
        if not possible and mark=='X':
            print("\nself position is already taken!")
        
        return possible
    
    
    # Determines whether a game winning condition exists
    # If so, returns True, and False otherwise
    def checkWin(self, mark):
        won = False # Variable holding the return value
        
        # Check wins by examining each combination of positions
        
        # Check each row
        for i in range(self.boardSize):
            won = True
            for j in range(self.boardSize):
                if self.marks[i][j]!=mark:
                    won=False
                    break        
            if won:
                break
        
        # Check each column
        if not won:
            for i in range(self.boardSize):
                won = True
                for j in range(self.boardSize):
                    if self.marks[j][i]!=mark:
                        won=False
                        break
                if won:
                    break

        # Check first diagonal
        if not won:
            for i in range(self.boardSize):
                won = True
                if self.marks[i][i]!=mark:
                    won=False
                    break
                
        # Check second diagonal
        if not won:
            for i in range(self.boardSize):
                won = True
                if self.marks[self.boardSize-1-i][i]!=mark:
                    won=False
                    break

        return won
    
    # Determines whether the board is full
    # If full, returns True, and False otherwise
    def noMoreMoves(self):
        return (self.marks!=' ').all()
  
    # function to make computer move and print the move the computer made
    def makeCompMove(self):
        state = State(self.marks.copy(), None)
        row, col  = alpha_beta(state).move
        
        row += 1
        col +=1

        print("Computer chose: "+str(row)+","+str(col))
        self.makeMove(row, col, 'O')

if __name__ == "__main__":
        
    # Print out the header info
    print("EXTRA CREDIT VERSION")
    print("CLASS: Artificial Intelligence, Lewis University")
    print("NAME: Prashamsh Takkalapally")

    LOST = 0
    WON = 1
    DRAW = 2    
    wrongInput = False
    boardSize = int(input("Please enter the size of the board n (e.g. n=3,4,5,...): "))
            
    # Create the game board of the given size
    board = GenGameBoard(boardSize)
            
    board.printBoard()  # Print the board before starting the game loop
            
    # Game loop
    while True:
        # *** Player's move ***        
        
        # Try to make the move and check if it was possible
        # If not possible get col,row inputs from player    
        row, col = -1, -1
        while not board.makeMove(row, col, 'X'):
            print("Player's Move")
            row, col = input("Choose your move (row, column): ").split(',')
            row = int(row)
            col = int(col)

        # Display the board again
        board.printBoard()
                
        # Check for ending condition
        # If game is over, check if player won and end the game
        if board.checkWin('X'):
            # Player won
            result = WON
            break
        elif board.noMoreMoves():
            # No moves left -> draw
            result = DRAW
            break
                
        # *** Computer's move ***
        board.makeCompMove()
        
        # Print out the board again
        board.printBoard()    
        
        # Check for ending condition
        # If game is over, check if computer won and end the game
        if board.checkWin('O'):
            # Computer won
            result = LOST
            break
        elif board.noMoreMoves():
            # No moves left -> draw
            result = DRAW
            break
            
    # Check the game result and print out the appropriate message
    print("GAME OVER")
    if result==WON:
        print("You Won!")            
    elif result==LOST:
        print("You Lost!")
    else: 
        print("It was a draw!")

