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
from langchain_experimental.tot.thought_generation import BaseThoughtGenerationStrategy
from langchain_experimental.tot.base import ToTChain
from langchain_core.prompts import PromptTemplate
from tic_tac_toe_checker import TicTacToeToTChecker
from loguru import logger


class TicTacToeCoTStrategy(BaseThoughtGenerationStrategy):
    """Custom CoT strategy for tic-tac-toe with A1-C3 coordinate system."""

    prompt: PromptTemplate = PromptTemplate(
        input_variables=["problem_description"],
        template="""You are an expert tic-tac-toe player. Your task is to analyze the current game state and make the best possible move.

Game Rules:
- The board is a 3x3 grid with positions numbered 1-9:
| 1 | 2 | 3 |
| 4 | 5 | 6 |
| 7 | 8 | 9 |

- Players take turns placing X or O
- The goal is to get 3 in a row (horizontally, vertically, or diagonally)
- If all 9 positions are filled without a winner, it's a draw

Winning Conditions:
- Horizontal: 1-2-3, 4-5-6, or 7-8-9
- Vertical: 1-4-7, 2-5-8, or 3-6-9
- Diagonal: 1-5-9 or 3-5-7

CRITICAL INSTRUCTIONS:
1. Look ONLY at the actual board state provided - do not imagine pieces that aren't there
2. Use 1-9 numbering system for all position references
3. Position 1 is top-left, position 5 is center, position 9 is bottom-right
4. Only consider moves to the available positions listed
5. A winning move requires 3 in a row (horizontal, vertical, or diagonal) - you need 2 pieces already in a line to win
6. Think about immediate winning opportunities first, then blocking opponent threats
7. The center (position 5) is typically the strongest opening move
8. Remember: You can only place ONE piece per turn - don't plan multiple moves ahead

Current Game State:
{problem_description}

Your task:
1. Carefully examine the EXACT board state provided above
2. Identify which positions are currently occupied and which are empty
3. List all available moves (only the positions that are actually empty)
4. For each available move, consider:
   - Does it create an immediate winning opportunity?
   - Does it block the opponent from winning on their next turn?
   - Does it control important strategic positions (center, corners)?
5. Select the best move based on this analysis
6. After your reasoning, always output your final move on a new line in this exact format:
Final Move: <position_number>

Think through this systematically and provide your final move as a position number (e.g., 1, 5, 9).""",
    )

    def next_thought(
        self, problem_description: str, thoughts_path: tuple = (), **kwargs
    ):
        """Generate the next thought using the custom prompt."""
        response_text = self.predict_and_parse(
            problem_description=problem_description, thoughts=thoughts_path, **kwargs
        )
        return response_text if isinstance(response_text, str) else ""


class TicTacToeCoTAgent:
    """
    Chain-of-Thought Agent for playing Tic-Tac-Toe.
    Uses linear reasoning to generate moves step by step.
    """

    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(api_key=api_key, model_name=model_name)

        # Note: Strategy and prompt are handled by ToTChain internally

        # Initialize ToTChain with custom CoT strategy
        self.chain = ToTChain(
            llm=self.llm,
            c=1,  # Linear reasoning - only one thought per step
            checker=TicTacToeToTChecker(),
            tot_strategy_class=TicTacToeCoTStrategy,
        )

        self.reasoning_trace = []

    def _format_game_state(self, state: Dict[str, Any]) -> str:
        """Format the game state for the LLM without coordinates."""
        board = state.get("board", [""] * 9)
        current_player = state.get("current_player", "O")

        # Create visual representation of the board
        board_str = ""
        for i in range(0, 9, 3):
            row = " | ".join([board[i + j] if board[i + j] else " " for j in range(3)])
            board_str += f"{row}\n"
            if i < 6:
                board_str += "_ _ _\n"

        # Available positions just by index (1–9)
        available_positions = [str(i + 1) for i, cell in enumerate(board) if not cell]

        return f"""Current Player: {current_player}
    
    {board_str}
    
    Available Positions: {", ".join(available_positions)}
    """

    def _parse_move_from_response(self, response: str) -> int:
        """Extract the move from the LLM response."""
        import re

        # First priority: Look for "Final Move: X" format (1-9 indexing)
        final_move_match = re.search(
            r"Final\s+Move\s*:\s*([1-9])", response, re.IGNORECASE
        )
        if final_move_match:
            move = int(final_move_match.group(1)) - 1  # Convert 1-9 to 0-8
            logger.info(f"Found 'Final Move' position: {move + 1}")
            return move

        # Second priority: Look for "I choose to place" format with 1-9 indexing
        choose_place_match = re.search(
            r"I\s+choose\s+to\s+place\s+[oO]\s+in\s+position\s+([1-9])",
            response,
            re.IGNORECASE,
        )
        if choose_place_match:
            move = int(choose_place_match.group(1)) - 1  # Convert 1-9 to 0-8
            logger.info(f"Found 'I choose to place' position: {move + 1}")
            return move

        # Third priority: Look for "Move to X" format (1-9 indexing)
        move_to_match = re.search(r"Move\s+to\s+([1-9])", response, re.IGNORECASE)
        if move_to_match:
            move = int(move_to_match.group(1)) - 1  # Convert 1-9 to 0-8
            logger.info(f"Found 'Move to' position: {move + 1}")
            return move

        # Fourth priority: Look for "position X" format (1-9 indexing)
        position_match = re.search(r"position\s+([1-9])", response, re.IGNORECASE)
        if position_match:
            move = int(position_match.group(1)) - 1  # Convert 1-9 to 0-8
            logger.info(f"Found 'position' reference: {move + 1}")
            return move

        # Fallback: Look for any number 1-9 in the response
        number_match = re.search(r"\b([1-9])\b", response)
        if number_match:
            move = int(number_match.group(1)) - 1  # Convert 1-9 to 0-8
            logger.info(f"Found number reference: {move + 1}")
            return move

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
