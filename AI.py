import math
import random
import numpy as np
import chess
import keras
from rich.console import Console

# Load pre-trained model
model = keras.models.load_model('best_chess_model.keras')

# Helper function to convert FEN to 3D tensor
def generate_bit_map(fen):
    board = chess.Board(fen)
    piece_map = {
        'k': 0, 'K': 0, 'q': 1, 'Q': 1, 'r': 2, 'R': 2,
        'n': 3, 'N': 3, 'b': 4, 'B': 4, 'p': 5, 'P': 5
    }
    material_values = {
        'k': 0, 'K': 0, 'q': -9, 'Q': 9, 'r': -5, 'R': 5,
        'n': -3, 'N': 3, 'b': -3, 'B': 3, 'p': -1, 'P': 1
    }
    bit_map = np.zeros((8, 8, 8))
    positions, _, _, _, _, _ = fen.split(' ')

    row, col = 0, 0
    white_material, black_material = 0, 0

    for char in positions:
        if char.isdigit():
            col += int(char)
        elif char == '/':
            row += 1
            col = 0
        else:
            layer = piece_map[char]
            bit_map[layer, row, col] = 1 if char.isupper() else -1
            if char.isupper():
                white_material += material_values[char]
            else:
                black_material += material_values[char]
            col += 1

    # Encode legal moves
    for move in board.legal_moves:
        to_square = chess.square_rank(move.to_square), chess.square_file(move.to_square)
        bit_map[6, to_square[0], to_square[1]] = 1 if board.turn else 0

    board.turn = not board.turn
    for move in board.legal_moves:
        to_square = chess.square_rank(move.to_square), chess.square_file(move.to_square)
        bit_map[7, to_square[0], to_square[1]] = 1 if board.turn else 0
    
    return bit_map

# Evaluate a board state using the trained model
def evaluate_board(fen):
    bit_map = generate_bit_map(fen)
    input_tensor = np.reshape(bit_map, (1, 1, 8, 8, 8))
    evaluation = model.predict(input_tensor, verbose=False)[0][0]
    return evaluation

# MCTS Node
class Node:
    def __init__(self, board, parent=None):
        self.board = board.copy()
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == len(list(self.board.legal_moves))

    def best_child(self, exploration_weight=1.0):
        choices_weights = [
            (child.value / (child.visits + 1e-6)) + exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

# MCTS Algorithm
def mcts(root, iterations):
    for _ in range(iterations):
        node = root
        # Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        # Expansion
        if not node.is_fully_expanded():
            legal_moves = list(node.board.legal_moves)
            for move in legal_moves:
                if all(child.board.peek() != move for child in node.children):
                    new_board = node.board.copy()
                    new_board.push(move)
                    child_node = Node(new_board, parent=node)
                    node.children.append(child_node)
                    node = child_node
                    break
        # Simulation
        board_copy = node.board.copy()
        for _ in range(10):  # Simulate random moves for 10 plies
            if board_copy.is_game_over():
                break
            random_move = random.choice(list(board_copy.legal_moves))
            board_copy.push(random_move)
        reward = evaluate_board(board_copy.fen())

        # Backpropagation
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent
    return root.best_child(exploration_weight=0).board.peek()

# Suggest best move for player
def suggest_move(board):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    best_move = None
    best_score = -float('inf') if board.turn else float('inf')

    for move in legal_moves:
        board.push(move)
        score = evaluate_board(board.fen())
        board.pop()

        if (board.turn and score > best_score) or (not board.turn and score < best_score):
            best_score = score
            best_move = move

    return best_move

# Main game loop with suggestion feature
def play_chess():
    board = chess.Board()
    console = Console()
    console.print("[green]Welcome to Chess AI with MCTS![/green]")
    # Choose to play as White or Black
    console.print("[yellow]Do you want to play as White or Black?[/yellow]")
    while True:
        choice = input("Enter 'W' for White or 'B' for Black: ").strip().upper()
        if choice == 'W':
            player_color = chess.WHITE
            break
        elif choice == 'B':
            player_color = chess.BLACK
            break
        else:
            console.print("[red]Invalid choice. Please enter 'W' or 'B'.[/red]")
    print("\n")
    print(board)
    print("\n")
    turn = 0

    def turn_player(board):
        while True:
            console.print("[yellow]Your move (Type 'hint' for a suggestion):[/yellow]")
            move = input("Enter your move: ").strip()

            if move.lower() == "hint":
                suggested_move = suggest_move(board)
                if suggested_move:
                    console.print(f"[green]Suggested move: {board.san(suggested_move)}[/green]")
                else:
                    console.print("[red]No suggestion available.[/red]")
                continue
            try:
                board.push_san(move)
                console.print(f"[green]You made a move: {move}[/green]")
                print("\n")
                print(board)
                print("\n")
                break  # Thoát khỏi vòng lặp sau khi người chơi thực hiện lượt
            except ValueError:
                console.print("[red]Invalid move. Try again.[/red]")
                continue

    def turn_ai(board):
        console.print("[yellow]AI is thinking...[/yellow]")
        root = Node(board)
        best_move = mcts(root, iterations=500)
        board.push(best_move)
        console.print(f"[bright_green]AI played: {best_move}[/bright_green]")
        print("\n")  
        print(board)
        print("\n")
    
    while not board.is_game_over():
        turn += 1
        console.print(f"[blue]Turn: {turn}[/blue]")

        if board.turn == player_color:
            turn_player(board)
            turn_ai(board)
        else:
            turn_ai(board)
            turn_player(board)
        
    console.print("[red]Game over![/red]")
    if board.is_checkmate():
        console.print("[red]Checkmate![/red]")
    elif board.is_stalemate():
        console.print("[yellow]Stalemate![/yellow]")
    elif board.is_insufficient_material():
        console.print("[yellow]Draw due to insufficient material![/yellow]")
    elif board.is_seventyfive_moves():
        console.print("[yellow]Draw due to 75-move rule![/yellow]")
    elif board.is_fivefold_repetition():
        console.print("[yellow]Draw due to fivefold repetition![/yellow]")

if __name__ == "__main__":
    play_chess()