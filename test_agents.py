"""
test_agents.py - Test script for Tic-Tac-Toe AI Agents

This script tests both CoT and ToT agents to ensure they work correctly.
"""

import os
from dotenv import load_dotenv
from agent_cot import create_cot_agent
from agent_tot import create_tot_agent
from tic_tac_toe import create_new_game
from tic_tac_toe_checker import TicTacToeToTChecker


def test_agents():
    """Test both agents with a simple game state."""
    load_dotenv()

    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found. Please set it in your environment.")
        return

    print("🧪 Testing Tic-Tac-Toe AI Agents")
    print("=" * 50)

    # Test the custom checker
    print("🔍 Testing Custom ToT Checker...")
    checker = TicTacToeToTChecker()

    # Test with a simple game state
    test_problem = """Current Player: O
Board State:
X |   |  
---------
  |   |  
---------
  |   |  
Available positions: [1, 2, 3, 4, 5, 6, 7, 8]"""

    test_thoughts = (
        "I should take the center position 4 to control the board",
        "Taking corner position 0 would be good for strategy",
        "Position 1 is not a good move as it's not strategic",
    )

    validity = checker.evaluate(test_problem, test_thoughts)
    print(f"✅ Checker evaluation result: {validity}")
    print()

    # Create a test game state
    game = create_new_game()

    # Test state: X has made one move, O needs to respond
    game.make_move(0, "X")  # X takes top-left corner

    test_state = game.get_state()
    print(f"Test game state: {test_state['board']}")
    print(f"Current player: {test_state['current_player']}")
    print()

    try:
        # Test CoT Agent
        print("🤖 Testing CoT Agent...")
        cot_agent = create_cot_agent()
        cot_response = cot_agent.run(test_state)

        print(f"✅ CoT Agent Response:")
        print(f"   Move: {cot_response['move']}")
        print(f"   Success: {cot_response['success']}")
        print(f"   Agent Type: {cot_response['agent_type']}")
        print(f"   Reasoning: {cot_response['reasoning_trace'][0]['thought'][:100]}...")
        print()

    except Exception as e:
        print(f"❌ CoT Agent Error: {e}")
        print()

    try:
        # Test ToT Agent
        print("🌳 Testing ToT Agent...")
        tot_agent = create_tot_agent()
        tot_response = tot_agent.run(test_state)

        print(f"✅ ToT Agent Response:")
        print(f"   Move: {tot_response['move']}")
        print(f"   Success: {tot_response['success']}")
        print(f"   Agent Type: {tot_response['agent_type']}")
        print(
            f"   Number of reasoning branches: {len(tot_response['reasoning_trace'])}"
        )
        print(
            f"   First branch: {tot_response['reasoning_trace'][0]['reasoning'][:100]}..."
        )
        print()

    except Exception as e:
        print(f"❌ ToT Agent Error: {e}")
        print()

    print("🎯 Test completed!")


if __name__ == "__main__":
    test_agents()
