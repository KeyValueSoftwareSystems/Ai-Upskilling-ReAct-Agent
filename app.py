# app.py

import streamlit as st

from agent_cot import create_cot_agent
from agent_tot import create_tot_agent
from tic_tac_toe import create_new_game
from dotenv import load_dotenv

list_of_agents = {
    "CoT Agent": create_cot_agent,
    "ToT Agent": create_tot_agent,
}

load_dotenv()

st.set_page_config(page_title="Tic-Tac-Toe AI Agents", layout="wide")

# --- Make Sidebar Wider ---
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            min-width: 400px;
            max-width: 400px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session State Initialization ---
if "game" not in st.session_state:
    st.session_state.game = create_new_game()

if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = "CoT Agent"

if "user_selected_agent" not in st.session_state:
    st.session_state.user_selected_agent = "CoT Agent"

if "agent_reasoning" not in st.session_state:
    st.session_state.agent_reasoning = None

if (
    "agent" not in st.session_state
    or st.session_state.selected_agent != st.session_state.user_selected_agent
):
    try:
        agent_factory = list_of_agents[st.session_state.user_selected_agent]
        st.session_state.agent = agent_factory()
        st.session_state.selected_agent = st.session_state.user_selected_agent
    except (ValueError, KeyError, AttributeError) as e:
        st.session_state.agent = None
        st.error(f"Failed to initialize {st.session_state.user_selected_agent}: {e}")


# --- Sidebar ---
with st.sidebar:
    st.header("🎮 Tic-Tac-Toe AI Agents")

    # Agent selection
    selected_agent = st.selectbox("Select Agent", list(list_of_agents.keys()))
    st.session_state.user_selected_agent = selected_agent

    st.divider()

    # Game controls
    st.subheader("Game Controls")
    if st.button("🔄 New Game"):
        st.session_state.game = create_new_game()
        st.session_state.agent_reasoning = None
        st.rerun()

    # Game status
    st.subheader("Game Status")
    game = st.session_state.game
    st.write(f"**Status:** {game.get_status_message()}")
    st.write(f"**Moves:** {len(game.move_history)}")

    st.divider()

    # Agent info
    st.subheader("Agent Info")
    if st.session_state.agent:
        agent_type = st.session_state.agent.__class__.__name__
        st.write(f"**Agent Type:** {agent_type}")
        st.write(
            f"**Strategy:** {'Chain-of-Thought' if 'CoT' in agent_type else 'Tree-of-Thought'}"
        )
    else:
        st.error("❌ Agent not initialized")


# --- Main Game Interface ---
st.title("🎮 Tic-Tac-Toe vs AI Agent")

# Create two columns: game board and reasoning
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎯 Game Board")

    # Display the game board
    game = st.session_state.game
    board = game.board

    # A1-C3 coordinate mapping
    coordinates = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]

    # Create 3x3 grid of buttons with coordinates
    for i in range(0, 9, 3):
        cols = st.columns(3)
        for j in range(3):
            idx = i + j
            coord = coordinates[idx]
            with cols[j]:
                if board[idx] == "":
                    if st.button(
                        f" \n{coord}", key=f"btn_{idx}", disabled=game.is_game_over()
                    ):
                        # Human player makes move
                        if game.make_move(idx, "X"):
                            st.session_state.agent_reasoning = None
                            st.rerun()
                else:
                    st.button(f"{board[idx]}\n{coord}", key=f"btn_{idx}", disabled=True)

    # Game status
    st.write(f"**{game.get_status_message()}**")

    # Show board as text with A1-C3 coordinates
    with st.expander("📋 Board State (Text)", expanded=False):
        # Create enhanced board display with coordinates
        board_str = ""
        for i in range(0, 9, 3):
            row = " | ".join([board[i + j] if board[i + j] else " " for j in range(3)])
            coord_row = " | ".join([coordinates[i + j] for j in range(3)])
            board_str += f"{row}\n"
            board_str += f"{coord_row}\n"
            if i < 6:
                board_str += "---------\n"
        st.text(board_str)

with col2:
    st.subheader("🧠 Agent Reasoning")

    if st.session_state.agent_reasoning:
        reasoning = st.session_state.agent_reasoning

        # Show agent type
        agent_type = reasoning.get("agent_type", "Unknown")
        st.write(f"**Agent:** {agent_type}")

        # Show reasoning trace
        reasoning_trace = reasoning.get("reasoning_trace", [])

        if agent_type == "CoT":
            # Chain-of-Thought display
            for i, step in enumerate(reasoning_trace):
                with st.expander(f"Step {i+1}: Chain-of-Thought", expanded=True):
                    st.write(step.get("thought", "No reasoning provided"))

        elif agent_type == "ToT":
            # Tree-of-Thought display
            for i, branch in enumerate(reasoning_trace):
                with st.expander(
                    f"Branch {branch.get('branch', i+1)}: Tree-of-Thought",
                    expanded=True,
                ):
                    st.write(branch.get("reasoning", "No reasoning provided"))

            # Show final reasoning
            final_reasoning = reasoning.get("final_reasoning", "")
            if final_reasoning:
                with st.expander("🎯 Final Decision", expanded=True):
                    st.write(final_reasoning)

        # Show move made
        move = reasoning.get("move", -1)
        if move >= 0:
            st.write(f"**Move Made:** Position {move}")

        # Show success status
        success = reasoning.get("success", False)
        if success:
            st.success("✅ Agent reasoning completed successfully")
        else:
            st.error("❌ Agent reasoning failed")

    else:
        st.info(
            "🤔 No reasoning available yet. Make a move to see the agent's thinking!"
        )

# Auto-make agent move after human move
if (
    not game.is_game_over()
    and game.current_player == "O"
    and st.session_state.agent
    and len(game.move_history) > 0
    and game.move_history[-1]["player"] == "X"
):

    with st.spinner("🤖 Agent is thinking..."):
        try:
            # Get agent's move
            game_state = game.get_state()
            response = st.session_state.agent.run(game_state)

            # Store reasoning for display
            st.session_state.agent_reasoning = response

            # Make the agent's move
            agent_move = response.get("move", 0)
            if game.make_move(agent_move, "O"):
                st.rerun()
            else:
                st.error("❌ Agent made an invalid move")

        except (ValueError, KeyError, AttributeError) as e:
            st.error(f"❌ Agent error: {e}")
            # Fallback: make a random valid move
            available_moves = game.get_available_moves()
            if available_moves:
                game.make_move(available_moves[0], "O")
                st.rerun()
