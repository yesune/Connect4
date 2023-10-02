from sklearn import svm
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import math
import time
import os
import warnings
import copy
import pdb

# just to see how long this code takes
start_time = time.time()

# read in the data
data = pd.read_csv("connect-4.data")

# convert data into numerical values
data.replace({'b': 0, 'x': 1, 'o': -1, 'win': 1, 'loss': -1, 'draw' : 0}, inplace=True)
# IGNORE THIS ITS JUST FOR TESTINGprint(data)

# preprocess the data
# preprocessing the data helps to improve the performance 
# and accuracy of the machine learning model by making the 
# data more consistent and suitable for the model to learn from
def preprocess(data):
    X = data.iloc[:, 0:42]  # all rows, all the features and no labels
    y = data.iloc[:, 42]  # all rows, label only

    return X, y

# splitting data into training and testing
all_X, all_y = preprocess(data)
X_train, X_test, y_train, y_test = train_test_split(all_X, all_y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

if os.path.exists('finalized_model.joblib'):
    clf = joblib.load('finalized_model.joblib')
    warnings.filterwarnings("ignore", category=UserWarning)
else:
    clf = svm.SVC(probability = True)
    print("begin training")
    clf.fit(X_train, y_train) 
    print("End training")

    print(clf.score(X_test, y_test))

    joblib.dump(clf, 'finalized_model.joblib')

elapsed_time = time.time() - start_time
print("Elapsed time: {} seconds".format(elapsed_time))

# Time to make the stuff for the game itself
gamestate = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
# Find out whose turn it is
user_turn = int(input("If you start, enter 1. Else, enter 0: "))
ai_started = user_turn == 0
turn = True
if user_turn == 0:
    turn = False

# print function
def print4(value):
    if value == 0:
        print('-', end = ' ')
    if value == 1:
        print('X', end = ' ')
    if value == -1:
        print('O', end = ' ')

# check for win
def gamedone(gamestate, player):
    # horizontal check
    for i in range(6): # height
        for j in range(7 - 3): # width
            if (gamestate[0][j * 6 + i] == player and gamestate[0][(j + 1) * 6 + i] == player 
            and gamestate[0][(j + 2) * 6 + i] == player and gamestate[0][(j + 3) * 6 + i] == player):
                return True
    # vertical check
        for j in range(7): # width
            for i in range(6 - 3): # height
                if (gamestate[0][j * 6 + i] == player and gamestate[0][j * 6 + i + 1] == player 
                and gamestate[0][j * 6 + i + 2] == player and gamestate[0][j * 6 + i + 3] == player):
                    return True
    # ascending diagonal check
        for j in range(7 - 3): # width
            for i in range(6 - 3): # height
                if (gamestate[0][j * 6 + i] == player and gamestate[0][(j + 1) * 6 + i + 1] == player 
                and gamestate[0][(j + 2) * 6 + i + 2] == player and gamestate[0][(j + 3) * 6 + i + 3] == player):
                    return True
    # descending diagonal check
        for j in range(7 - 3): # width
            for i in range(3, 7): # height
                if (gamestate[0][j * 6 + i] == player and gamestate[0][(j + 1) * 6 + i - 1] == player 
                and gamestate[0][(j + 2) * 6 + i - 2] == player and gamestate[0][(j + 3) * 6 + i - 3] == player):
                    return True

    #   if (this.board[i][j] == player && this.board[i][j+1] == player && this.board[i][j+2] == player && this.board[i][j+3] == player){

# add piece function
def addpiece(user_input, gamestate, char):
    count = 0
    while count < 6:
        if gamestate[0][user_input * 6 - 6 + count] == 0:
            gamestate[0][user_input * 6 - 6 + count] = char
            break
        else:
            count += 1
    if count == 6:
        return False
    else:
        return True

def removePiece(user_input, gamestate):
    for i in range(6, 0, -1):
        if gamestate[0][user_input * 6 - 6 + i - 1] != 0:
            gamestate[0][user_input * 6 - 6 + i - 1] = 0
            return True
    return False

# Simulating future gamestates to see which move is the best
def predict_best_move(state, depth):
    if depth == 0:
        return clf.predict(state)[0]
        
    good_column = [0] * 7

    # best_score = -math.inf if max_player else math.inf
    for i in range(1, 8):
        if not addpiece(i, state, 1):
            good_column[i-1] = -math.inf
            continue
        if gamedone(state, 1):
            return i -1
        for j in range(1, 8):
            if not addpiece(j, state, -1):
                continue
            if gamedone(state, -1): # other player can win off this
                good_column[i-1] = -100000
                removePiece(j, state)
                continue
            for k in range(1, 8):
                #if good_column[i-1] < -100:
                #    continue
                if not addpiece(k, state, 1):
                    continue
                if gamedone(state, 1):
                    print("game done", j, k)
                    for b in range(6):
                        for a in range(7):
                            print4(state[0][6 * a + 6 - b - 1])
                        print("")
                    print("-------------")
                    print("1 2 3 4 5 6 7")
                    freeWin = True
                    removePiece(k, state)
                    removePiece(j, state)
                    for n in range(1, 8):
                        if not addpiece(n, state, -1):
                            continue
                        hasWin = False
                        for m in range(1, 8):
                            if not addpiece(m, state, 1):
                                continue
                            if gamedone(state, 1):
                                hasWin = True
                                print(n, m)
                                removePiece(m, state)
                                break
                            removePiece(m, state)
                        if not hasWin:
                            freeWin = False
                            removePiece(n, state)
                            break
                        removePiece(n, state)
                    if freeWin:
                        return i - 1
                    addpiece(j, state, -1)
                    addpiece(k, state, 1)

                for l in range(1, 8):
                    if not addpiece(l, state, -1):
                        continue
                    score = clf.predict(state)[0]
                    if gamedone(state, -1):
                        score = -100000
                    good_column[i - 1] += score
                    removePiece(l, state)
                removePiece(k, state)
            removePiece(j, state)
        removePiece(i, state)

    print("My predictions", good_column)
    return good_column.index(max(good_column)) + 1

def predict_best_move_rec(state, depth, playerStarted):
    if depth != 0:
        print(depth)
        char = -1
        if not playerStarted:
            depth -= 1
            char = 1
        else:
            test = clf.predict(state)[0]
            if test == -1:
                return -1, 3
        a = []
        for i in range(1, 8):
            success = addpiece(i, state, char)
            if success:
                if gamedone(state, i):
                    a.append(1)
                score, _ = predict_best_move_rec(state, depth, not playerStarted)
                a.append(score)
            else:
                a.append(0)
            removePiece(i, state)
        print(a)
        return sum(a), a.index(max(a)) + 1
    else:
        return clf.predict(state)[0], -1

def minimax(state, depth, alpha, beta, maximizingPlayer):
    if depth == 0:
        return get_score(state), 3
    if gamedone(state, 1):
        return math.inf, 3
    if gamedone(state, -1):
        return -1 * math.inf, 3
    
    if maximizingPlayer:
        best_move = -1
        max_eval = -1 * math.inf
        for i in range (1, 8):
            temp_state = copy.deepcopy(state)
            if addpiece(i, temp_state, 1):
                eval, _ = minimax(temp_state, depth - 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_move = i
                alpha = max(alpha, max_eval)
                if beta <= alpha:
                    break
        return max_eval, best_move

    else:  # Minimizing player (opponent)
        best_move = -1
        min_eval = math.inf
        for i in range(1, 8):
            temp_state = copy.deepcopy(state)
            if addpiece(i, temp_state, -1):
                eval_score, _ = minimax(temp_state, depth - 1, alpha, beta, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = i
                beta = min(beta, min_eval)
                if beta <= alpha:
                    break
        return min_eval, best_move

def get_score(state):
    res = clf.predict_proba(state)
    return res[0][2] - res[0][0]

# Takes in user input, adds it to gamestate, predicts best move
while True:

    print("\n")
    # predict
    if turn == False:
        temp_gamestate = copy.deepcopy(gamestate)
        _, best_move = minimax(temp_gamestate, 6, -math.inf, math.inf, True)
        print("I went", best_move)
        addpiece(best_move, gamestate, 1)
        if gamedone(gamestate, 1):
            for j in range(6):
                for i in range(7):
                    print4(gamestate[0][6 * i + 6 - j - 1])
                print("")
            print("-------------")
            print("1 2 3 4 5 6 7")
            print("Player X won")
            exit()
    else:
        # print out what the gamestate looks like
        for j in range(6):
            for i in range(7):
                print4(gamestate[0][6 * i + 6 - j - 1])
            print("")
        print("-------------")
        print("1 2 3 4 5 6 7")

        # check who's turn
        char = -1

        # ask for user input
        user_input = int(input("Enter a position:"))
        
        # exit condition
        if user_input == -1:
            break

        if user_input > 7 or user_input < 1:
            continue

        # add the piece
        if not addpiece(user_input, gamestate, char):
            print("Column is full!")
            continue

        # check for win
        if gamedone(gamestate, -1):
            print("Player O won")
            exit()

    # swap the piece
    turn = not turn