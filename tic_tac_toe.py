"""
tic_tac_toe.py - Tic-Tac-Toe Game Logic

This file contains:
- Game state management
- Win condition checking
- Move validation
- Game status tracking
"""

from typing import Dict, Any, List, Optional
from enum import Enum


class GameStatus(Enum):
    """Enum for game status."""

    ONGOING = "ongoing"
    X_WINS = "x_wins"
    O_WINS = "o_wins"
    DRAW = "draw"


class TicTacToeGame:
    """
    Tic-Tac-Toe game logic and state management.
    """

    def __init__(self):
        self.board = [""] * 9  # 3x3 board represented as 1D array
        self.current_player = "X"  # X always starts
        self.game_status = GameStatus.ONGOING
        self.move_history = []

    def reset_game(self):
        """Reset the game to initial state."""
        self.board = [""] * 9
        self.current_player = "X"
        self.game_status = GameStatus.ONGOING
        self.move_history = []

    def get_state(self) -> Dict[str, Any]:
        """Get current game state."""
        return {
            "board": self.board.copy(),
            "current_player": self.current_player,
            "game_status": self.game_status.value,
            "move_history": self.move_history.copy(),
        }

    def is_valid_move(self, position: int) -> bool:
        """Check if a move is valid."""
        return (
            0 <= position <= 8
            and self.board[position] == ""
            and self.game_status == GameStatus.ONGOING
        )

    def make_move(self, position: int, player: str = None) -> bool:
        """
        Make a move on the board.

        Args:
            position: Position to place the mark (0-8)
            player: Player making the move (optional, uses current_player if not provided)

        Returns:
            True if move was successful, False otherwise
        """
        if player is None:
            player = self.current_player

        if not self.is_valid_move(position):
            return False

        # Make the move
        self.board[position] = player
        self.move_history.append({"position": position, "player": player})

        # Check for win or draw
        self._check_game_status()

        # Switch player if game is still ongoing
        if self.game_status == GameStatus.ONGOING:
            self.current_player = "O" if self.current_player == "X" else "X"

        return True

    def _check_game_status(self):
        """Check if the game has ended and update status."""
        # Check for win conditions
        win_conditions = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],  # Rows
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],  # Columns
            [0, 4, 8],
            [2, 4, 6],  # Diagonals
        ]

        for condition in win_conditions:
            if (
                self.board[condition[0]]
                == self.board[condition[1]]
                == self.board[condition[2]]
                != ""
            ):
                if self.board[condition[0]] == "X":
                    self.game_status = GameStatus.X_WINS
                else:
                    self.game_status = GameStatus.O_WINS
                return

        # Check for draw
        if all(cell != "" for cell in self.board):
            self.game_status = GameStatus.DRAW

    def get_available_moves(self) -> List[int]:
        """Get list of available move positions."""
        return [i for i, cell in enumerate(self.board) if cell == ""]

    def get_winner(self) -> Optional[str]:
        """Get the winner if game is over."""
        if self.game_status == GameStatus.X_WINS:
            return "X"
        elif self.game_status == GameStatus.O_WINS:
            return "O"
        return None

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.game_status != GameStatus.ONGOING

    def get_board_display(self) -> str:
        """Get a string representation of the board."""
        display = ""
        for i in range(0, 9, 3):
            row = " | ".join(
                [self.board[i + j] if self.board[i + j] else " " for j in range(3)]
            )
            display += f"{row}\n"
            if i < 6:
                display += "---------\n"
        return display

    def get_status_message(self) -> str:
        """Get a human-readable status message."""
        if self.game_status == GameStatus.ONGOING:
            return f"Current player: {self.current_player}"
        elif self.game_status == GameStatus.X_WINS:
            return "X wins!"
        elif self.game_status == GameStatus.O_WINS:
            return "O wins!"
        else:
            return "It's a draw!"


def create_new_game() -> TicTacToeGame:
    """Create a new tic-tac-toe game."""
    return TicTacToeGame()
