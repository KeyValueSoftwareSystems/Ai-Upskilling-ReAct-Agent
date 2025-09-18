"""
agent_cot.py - Chain-of-Thought Agent for Tic-Tac-Toe

This file contains:
- CoT agent class using SampleCoTStrategy and ToTChain
- Tic-tac-toe specific reasoning and move generation
- Integration with LangChain Experimental ToT framework
"""

import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_experimental.tot.thought_generation import SampleCoTStrategy
from langchain_experimental.tot.base import ToTChain
from tic_tac_toe_checker import TicTacToeToTChecker
from loguru import logger


class TicTacToeCoTAgent:
    """
    Chain-of-Thought Agent for playing Tic-Tac-Toe.
    Uses linear reasoning to generate moves step by step.
    """

    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(api_key=api_key, model_name=model_name)

        # Note: Strategy and prompt are handled by ToTChain internally

        # Initialize ToTChain with CoT strategy
        self.chain = ToTChain(
            llm=self.llm,
            c=1,  # Linear reasoning - only one thought per step
            checker=TicTacToeToTChecker(),
            tot_strategy_class=SampleCoTStrategy,
        )

        self.reasoning_trace = []

    def _create_tic_tac_toe_prompt(self) -> str:
        """Create a custom prompt for tic-tac-toe reasoning."""
        return """You are an expert tic-tac-toe player. Your task is to analyze the current game state and make the best possible move.

Game Rules:
- The board is a 3x3 grid with positions identified by coordinates:
| A1 | A2 | A3 |
| B1 | B2 | B3 |
| C1 | C2 | C3 |
- Players take turns placing X or O
- The goal is to get 3 in a row (horizontally, vertically, or diagonally)
- If all 9 positions are filled without a winner, it's a draw

Winning Conditions:
- Horizontal: A1-A2-A3, B1-B2-B3, or C1-C2-C3
- Vertical: A1-B1-C1, A2-B2-C2, or A3-B3-C3
- Diagonal: A1-B2-C3 or A3-B2-C1

IMPORTANT: Use A1-C3 coordinate system for all position references. A1 is top-left, B2 is center, C3 is bottom-right.

Current Game State:
{problem_description}

Your task:
1. Analyze the current board state
2. Identify all possible moves (using A1-C3 coordinates)
3. Consider the consequences of each move
4. Select the best move that maximizes your chance of winning or prevents the opponent from winning
5. Provide your reasoning step by step

Think through this systematically and provide your final move as a coordinate (e.g., A1, B2, C3)."""

    def _format_game_state(self, state: Dict[str, Any]) -> str:
        """Format the game state for the LLM."""
        board = state.get("board", [""] * 9)
        current_player = state.get("current_player", "O")

        # Create visual representation of the board with A1-C3 coordinates
        coordinates = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]

        board_str = ""
        for i in range(0, 9, 3):
            row = " | ".join([board[i + j] if board[i + j] else " " for j in range(3)])
            coord_row = " | ".join([coordinates[i + j] for j in range(3)])
            board_str += f"{row}\n"
            board_str += f"{coord_row}\n"
            if i < 6:
                board_str += "---------\n"

        # Convert available positions to A1-C3 format
        available_positions = [
            coordinates[i] for i, cell in enumerate(board) if not cell
        ]
        return f"""Current Player: {current_player}
Board State (with A1-C3 coordinates):
{board_str}
Available positions (A1-C3 coordinates): {available_positions}"""

    def _parse_move_from_response(self, response: str) -> int:
        """Extract the move from the LLM response."""
        import re

        # Coordinate mapping: A1-C3 to 0-8 indices
        coord_to_index = {
            "A1": 0,
            "A2": 1,
            "A3": 2,
            "B1": 3,
            "B2": 4,
            "B3": 5,
            "C1": 6,
            "C2": 7,
            "C3": 8,
        }

        # Look for explicit coordinate statements first (highest priority)
        coord_patterns = [
            r"place\s+[oO]\s+(?:at\s+)?([A-C][1-3])",
            r"move\s+[oO]\s+to\s+([A-C][1-3])",
            r"([A-C][1-3])\s+is\s+the\s+best",
            r"choose\s+([A-C][1-3])",
            r"select\s+([A-C][1-3])",
            r"let's\s+place\s+[oO]\s+at\s+([A-C][1-3])",
            r"placing\s+[oO]\s+at\s+([A-C][1-3])",
            r"play\s+in\s+the\s+center.*?([A-C][1-3])",
            r"center.*?([A-C][1-3])",
            r"best\s+move\s+is.*?([A-C][1-3])",
            r"will\s+play.*?([A-C][1-3])",
            r"final\s+move.*?([A-C][1-3])",
            r"decision.*?([A-C][1-3])",
            r"conclusion.*?([A-C][1-3])",
            r"strategic.*?([A-C][1-3])",
        ]

        for pattern in coord_patterns:
            match = re.search(pattern, response.upper())
            if match:
                coord = match.group(1)
                if coord in coord_to_index:
                    return coord_to_index[coord]

        # Look for standalone coordinates
        coord_matches = re.findall(r"\b([A-C][1-3])\b", response.upper())
        if coord_matches:
            # Take the last coordinate mentioned
            last_coord = coord_matches[-1]
            if last_coord in coord_to_index:
                return coord_to_index[last_coord]

        # Fallback: look for any coordinate pattern
        coord_match = re.search(r"([A-C][1-3])", response.upper())
        if coord_match:
            coord = coord_match.group(1)
            if coord in coord_to_index:
                return coord_to_index[coord]

        # Default fallback (center position)
        return 4

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the CoT agent on the given game state.

        Args:
            state: Dictionary containing 'board' and 'current_player'

        Returns:
            Dictionary with 'move', 'reasoning_trace', and 'success'
        """
        self.reasoning_trace = []

        try:
            # Format the game state
            formatted_state = self._format_game_state(state)

            # Run the chain
            logger.info(f"Running CoT agent with state: {formatted_state}")
            output = self.chain.invoke({"problem_description": formatted_state})

            # Extract the final answer
            final_answer = output.get("response", "")
            logger.info(f"CoT Agent output: {final_answer}")

            # Parse the move
            move = self._parse_move_from_response(final_answer)

            # Validate the move
            board = state.get("board", [""] * 9)
            if move < 0 or move > 8 or board[move] != "":
                # Find first available move as fallback
                for i in range(9):
                    if board[i] == "":
                        move = i
                        break
            logger.info(f"CoT Agent move: {move}")
            return {
                "move": move,
                "reasoning_trace": [{"step": 1, "thought": final_answer}],
                "success": True,
                "agent_type": "CoT",
            }

        except (ValueError, KeyError, AttributeError) as e:
            logger.error(f"Error in CoT agent: {e}")
            # Fallback: return first available move
            board = state.get("board", [""] * 9)
            for i in range(9):
                if board[i] == "":
                    return {
                        "move": i,
                        "reasoning_trace": [
                            {
                                "step": 1,
                                "thought": f"Error occurred, using fallback move: {e}",
                            }
                        ],
                        "success": False,
                        "agent_type": "CoT",
                    }

            return {
                "move": 0,
                "reasoning_trace": [{"step": 1, "thought": "No valid moves available"}],
                "success": False,
                "agent_type": "CoT",
            }


def create_cot_agent() -> TicTacToeCoTAgent:
    """
    Create and return a CoT agent instance.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return TicTacToeCoTAgent(api_key)
