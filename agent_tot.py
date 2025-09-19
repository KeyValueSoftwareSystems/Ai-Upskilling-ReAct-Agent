"""
agent_tot.py - Tree-of-Thought Agent for Tic-Tac-Toe

This file contains:
- ToT agent class using TreeofThoughtStrategy and ToTChain
- Tic-tac-toe specific reasoning with multiple branch exploration
- Integration with LangChain Experimental ToT framework
"""

import os
from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain_experimental.tot.thought_generation import ProposePromptStrategy, BaseThoughtGenerationStrategy
from langchain_experimental.tot.base import ToTChain
from langchain_core.prompts import PromptTemplate
from tic_tac_toe_checker import TicTacToeToTChecker
from loguru import logger


class TicTacToeToTStrategy(BaseThoughtGenerationStrategy):
    """Custom ToT strategy for tic-tac-toe with A1-C3 coordinate system."""

    prompt: PromptTemplate = PromptTemplate(
        input_variables=["problem_description"],
        template="""You are an expert tic-tac-toe player using Tree-of-Thought reasoning. Your task is to analyze the current game state and explore multiple possible moves before selecting the best one.

Game Rules:
- The board is a 3x3 grid with positions numbered 1-9:
| 1 | 2 | 3 |
| 4 | 5 | 6 |
| 7 | 8 | 9 |

- Players take turns placing X or O
- The goal is to get 3 in a row (horizontally, vertically, or diagonally) of the same player
- the 3 in a row can be horizontal, vertical, or diagonal of the same player
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
3. Generate multiple candidate moves (at least 3 different options using only available positions)
4. For each candidate move, think through the potential consequences:
   - Does it create an immediate winning opportunity?
   - Does it block the opponent from winning on their next turn?
   - Does it control important strategic positions (center, corners)?
5. Consider both offensive and defensive strategies
6. Evaluate the strength of each move based on:
   - Immediate winning opportunities
   - Blocking opponent's winning opportunities
   - Creating future winning opportunities
   - Center control and corner advantages
7. Select the best move after considering all alternatives
8. After your reasoning, if you have selected a move and you are confident about your choice, always output your final move on a new line in this exact format:
Final Move: <position_number>

Think through this systematically, exploring multiple branches of reasoning, and provide your final move as a position number (e.g., 1, 5, 9) along with your reasoning for why this move is superior to the alternatives.""",
    )

    def next_thought(
        self, problem_description: str, thoughts_path: tuple = (), **kwargs
    ):
        """Generate the next thought using the custom prompt."""
        response_text = self.predict_and_parse(
            problem_description=problem_description, thoughts=thoughts_path, **kwargs
        )
        return response_text if isinstance(response_text, str) else ""


class TicTacToeToTAgent:
    """
    Tree-of-Thought Agent for playing Tic-Tac-Toe.
    Explores multiple branches of possible moves before selecting the best one.
    """

    def __init__(self, api_key: str, model_name: str = "openai/gpt-oss-120b"):
        self.llm = ChatGroq(api_key=api_key, model_name=model_name)

        # Initialize ToTChain with custom ToT strategy
        # Note: TicTacToeToTStrategy already has its own prompt defined
        self.chain = ToTChain(
            llm=self.llm,
            c=3,  # Explore multiple branches - 3 thoughts per step
            checker=TicTacToeToTChecker(),
            tot_strategy_class=TicTacToeToTStrategy,
        )

        self.reasoning_trace = []

    def _create_tic_tac_toe_prompt(self) -> str:
        """Create a custom prompt for tic-tac-toe reasoning with tree exploration."""
        return """You are an expert tic-tac-toe player using Tree-of-Thought reasoning. Your task is to analyze the current game state and explore multiple possible moves before selecting the best one.

Game Rules:
- The board is a 3x3 grid with positions identified by coordinates:
| A1 | A2 | A3 |
| B1 | B2 | B3 |
| C1 | C2 | C3 |

- Players take turns placing X or O
- Your goal is to get 3 O characters in a winning condition (horizontally, vertically, or diagonally) while blocking the player from getting 3 X 's in one of the winning condition.
- If all 9 positions are filled without a winner, it's a draw

Winning Conditions for O and X:
- Horizontal: A1-A2-A3, B1-B2-B3, or C1-C2-C3
- Vertical: A1-B1-C1, A2-B2-C2, or A3-B3-C3
- Diagonal: A1-B2-C3 or A3-B2-C1

IMPORTANT: Use A1-C3 coordinate system for all position references. A1 is top-left, B2 is center, C3 is bottom-right.

Current Game State:
{problem_description}

Your task:
1. Analyze the current board state
2. Generate multiple candidate moves (at least 3 different options using A1-C3 coordinates)
3. For each candidate move, think through the potential consequences
4. Consider both offensive and defensive strategies
5. Evaluate the strength of each move based on:
   - Immediate winning opportunities
   - Blocking opponent's winning opportunities
   - Creating future winning opportunities
   - Center control and corner advantages
6. Select the best move after considering all alternatives
7. After selecting the best move, always output your final move on a new line in this exact format:
Final Move: <coordinate>

Example:
Thought 1: ...your reasoning...
Thought 2: ...your reasoning...
..
Final Move: B2

Think through this systematically, exploring multiple branches of reasoning, and provide your final move as a coordinate (e.g., A1, B2, C3) along with your reasoning for why this move is superior to the alternatives. Make sure the Final Move is on its own line."""

    def _format_game_state(self, state: Dict[str, Any]) -> str:
        """Format the game state for the LLM."""
        board = state.get("board", [""] * 9)
        current_player = state.get("current_player", "O")
        
        coordinates = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]

        # Create visual representation of the board
        board_str = ""
        for i in range(0, 9, 3):
            row = " | ".join([board[i + j] if board[i + j] else " " for j in range(3)])
            board_str += f"{row}\n"
            if i < 6:
                board_str += "_ _ _\n"

        # Convert available positions to A1-C3 format
        available_positions = [
            coordinates[i] for i, cell in enumerate(board) if not cell
        ]
        
        game_state = f"""Current Player: {current_player}
Board State (with A1-C3 coordinates):
{board_str}
Available positions (A1-C3 coordinates): {available_positions}"""
        return game_state

    def _parse_move_from_response(self, response: str) -> int:
        """Extract the move from the LLM response."""
        import re
        logger.info(f"Parsing response: {response}")
        logger.debug(f"Response format check - contains '**Final Move:**'?: {'**Final Move:**' in response}")
        logger.debug(f"Response format check - contains 'Final Move:'?: {'Final Move:' in response}")
        
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
        
        # HIGHEST priority: Look for "Final Move: X" or "Final Move: XX" with bolded/starred format
        # This handles cases like "**Final Move:** 2" with special formatting
        special_final_move = re.search(r"\*{0,2}Final\s+Move:?\*{0,2}\s*:?\s*\*{0,2}\s*([1-9]|[A-C][1-3])\*{0,2}", response, re.IGNORECASE)
        if special_final_move:
            move_text = special_final_move.group(1)
            logger.info(f"Found specially formatted final move: {move_text}")
            
            # Check if it's an A1-C3 coordinate
            if re.match(r"[A-C][1-3]", move_text, re.IGNORECASE):
                coord = move_text.upper()
                if coord in coord_to_index:
                    logger.info(f"Parsed as A1-C3 coordinate: {coord}")
                    return coord_to_index[coord]
            
            # Check if it's a 1-9 digit
            if re.match(r"[1-9]", move_text):
                move = int(move_text) - 1  # Convert 1-9 to 0-8
                logger.info(f"Parsed as position number: {move + 1}")
                return move
        
        # Part 1: Look for A1-C3 coordinate format (primary format)
        # First priority: Look for the "Final Move: XX" format with A1-C3 coordinates
        final_move_match = re.search(r"Final\s+Move\s*:\s*([A-C][1-3])", response, re.IGNORECASE)
        if final_move_match:
            coord = final_move_match.group(1).upper()
            if coord in coord_to_index:
                logger.info(f"Found 'Final Move' coordinate: {coord}")
                return coord_to_index[coord]

        # Part 2: Look for 1-9 index format (secondary format)
        # Look for "Final Move: X" format (1-9 indexing)
        final_move_match = re.search(
            r"Final\s+Move\s*:\s*([1-9])", response, re.IGNORECASE
        )
        if final_move_match:
            move = int(final_move_match.group(1)) - 1  # Convert 1-9 to 0-8
            logger.info(f"Found 'Final Move' position: {move + 1}")
            return move
            
        # Look for explicit coordinate statements (third priority)
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
                    logger.info(f"Found coordinate: {coord}")
                    return coord_to_index[coord]

        # Look for standalone coordinates
        coord_matches = re.findall(r"\b([A-C][1-3])\b", response.upper())
        if coord_matches:
            # Take the last coordinate mentioned
            last_coord = coord_matches[-1]
            if last_coord in coord_to_index:
                logger.info(f"Found standalone coordinate: {last_coord}")
                return coord_to_index[last_coord]

        # Fallback: look for any coordinate pattern
        coord_match = re.search(r"([A-C][1-3])", response.upper())
        if coord_match:
            coord = coord_match.group(1)
            if coord in coord_to_index:
                logger.info(f"Found coordinate in fallback: {coord}")
                return coord_to_index[coord]
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

    def _extract_reasoning_branches(self, response: str) -> List[Dict[str, str]]:
        """Extract multiple reasoning branches from the ToT response."""
        branches = []

        # Try to split by common branch indicators
        branch_indicators = [
            "Branch 1:",
            "Option 1:",
            "Move 1:",
            "First option:",
            "Branch 2:",
            "Option 2:",
            "Move 2:",
            "Second option:",
            "Branch 3:",
            "Option 3:",
            "Move 3:",
            "Third option:",
            "Alternative 1:",
            "Alternative 2:",
            "Alternative 3:",
        ]

        current_branch = ""
        branch_count = 0

        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if any(indicator in line for indicator in branch_indicators):
                if current_branch and branch_count > 0:
                    branches.append(
                        {"branch": branch_count, "reasoning": current_branch.strip()}
                    )
                current_branch = line
                branch_count += 1
            elif current_branch:
                current_branch += f"\n{line}"

        # Add the last branch
        if current_branch and branch_count > 0:
            branches.append(
                {"branch": branch_count, "reasoning": current_branch.strip()}
            )

        # If no branches found, treat the whole response as one branch
        if not branches:
            branches.append({"branch": 1, "reasoning": response})

        return branches

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the ToT agent on the given game state.

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
            logger.info(f"Running ToT agent with state: {formatted_state}")
            output = self.chain.invoke({"problem_description": formatted_state})

            # Extract the final answer
            final_answer = output.get("response", "")
            logger.info(f"ToT Agent output: {final_answer}")

            # Parse the move
            move = self._parse_move_from_response(final_answer)

            # Extract reasoning branches
            reasoning_branches = self._extract_reasoning_branches(final_answer)
            logger.info(f"Extracted reasoning branches: {reasoning_branches}")

            # Validate the move
            board = state.get("board", [""] * 9)
            if move < 0 or move > 8 or board[move] != "":
                # Find first available move as fallback
                for i in range(9):
                    if board[i] == "":
                        move = i
                        break

            return {
                "move": move,
                "reasoning_trace": reasoning_branches,
                "success": True,
                "agent_type": "ToT",
                "final_reasoning": final_answer,
            }

        except (ValueError, KeyError, AttributeError) as e:
            logger.error(f"Error in ToT agent: {e}")
            # Fallback: return first available move
            board = state.get("board", [""] * 9)
            for i in range(9):
                if board[i] == "":
                    return {
                        "move": i,
                        "reasoning_trace": [
                            {
                                "branch": 1,
                                "reasoning": f"Error occurred, using fallback move: {e}",
                            }
                        ],
                        "success": False,
                        "agent_type": "ToT",
                        "final_reasoning": f"Error: {e}",
                    }

            return {
                "move": 0,
                "reasoning_trace": [
                    {"branch": 1, "reasoning": "No valid moves available"}
                ],
                "success": False,
                "agent_type": "ToT",
                "final_reasoning": "No valid moves available",
            }


def create_tot_agent() -> TicTacToeToTAgent:
    """
    Create and return a ToT agent instance.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    return TicTacToeToTAgent(api_key)
