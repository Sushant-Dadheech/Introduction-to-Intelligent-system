"""
Minimax Algorithm - Tic-Tac-Toe
An artificial intelligence that never loses at Tic-Tac-Toe, 
implemented using the classic Minimax adversarial search algorithm.
"""

import time
import sys
import math

def typing_print(text, delay=0.03, newline=True):
    """Outputs text with a typewriter effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    if newline:
        print()

def print_board(board):
    print("\n   0   1   2")
    print(f" 0 {board[0][0]} | {board[0][1]} | {board[0][2]} ")
    print("  ---+---+---")
    print(f" 1 {board[1][0]} | {board[1][1]} | {board[1][2]} ")
    print("  ---+---+---")
    print(f" 2 {board[2][0]} | {board[2][1]} | {board[2][2]} \n")

def check_winner(board):
    # Check rows and columns
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] != ' ':
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] and board[0][i] != ' ':
            return board[0][i]
    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != ' ':
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != ' ':
        return board[0][2]
    # Check tie
    for row in board:
        if ' ' in row:
            return None
    return 'Tie'

def minimax(board, depth, is_maximizing):
    result = check_winner(board)
    if result == 'X':
        return 10 - depth
    elif result == 'O':
        return -10 + depth
    elif result == 'Tie':
        return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'X'
                    score = minimax(board, depth + 1, False)
                    board[i][j] = ' '
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O'
                    score = minimax(board, depth + 1, True)
                    board[i][j] = ' '
                    best_score = min(score, best_score)
        return best_score

def ai_move(board):
    best_score = -math.inf
    move = None
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'X'  # AI is 'X'
                score = minimax(board, 0, False)
                board[i][j] = ' '
                if score > best_score:
                    best_score = score
                    move = (i, j)
    if move:
        board[move[0]][move[1]] = 'X'

def main():
    typing_print("=== 🤖 Minimax Algorithm: Unbeatable Tic-Tac-Toe ===", delay=0.04)
    typing_print("The AI uses Minimax to explore all possible game states.", delay=0.03)
    typing_print("You are 'O' and the AI is 'X'. AI goes first!\n", delay=0.03)
    
    board = [[' ' for _ in range(3)] for _ in range(3)]
    
    ai_move(board)  # AI makes the first move to save time
    
    while True:
        print_board(board)
        result = check_winner(board)
        if result:
            break
            
        # Player move
        valid_move = False
        while not valid_move:
            try:
                typing_print("👉 Enter your move (row and column, e.g. 1 1): ", delay=0.02, newline=False)
                row, col = map(int, input().split())
                if 0 <= row <= 2 and 0 <= col <= 2 and board[row][col] == ' ':
                    board[row][col] = 'O'
                    valid_move = True
                else:
                    print("❌ Invalid move! Cell is occupied or out of bounds.")
            except ValueError:
                print("❌ Invalid input! Please enter two numbers separated by a space.")
                
        result = check_winner(board)
        if result:
            print_board(board)
            break
            
        print("\nThinking...")
        time.sleep(0.5)
        ai_move(board)

    if result == 'Tie':
        typing_print("\n🤝 It's a Tie! (As expected when playing perfectly against Minimax)", delay=0.04)
    else:
        typing_print(f"\n🏆 The winner is: {result}", delay=0.05)
        if result == 'X':
            typing_print("🤖 AI wins! Minimax proves its dominance.", delay=0.04)
        else:
            typing_print("😲 You won?! The universe is broken.", delay=0.04)

if __name__ == "__main__":
    main()
