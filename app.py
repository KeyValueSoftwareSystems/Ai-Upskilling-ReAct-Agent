# app.py

import streamlit as st
import pandas as pd
from agent import create_agent as create_agent_main
from agent_checkpoints.agent_checkpoint_1 import (
    create_agent as create_agent_checkpoint_1,
)
from agent_checkpoints.agent_checkpoint_2 import (
    create_agent as create_agent_checkpoint_2,
)
from agent_checkpoints.agent_checkpoint_3 import (
    create_agent as create_agent_checkpoint_3,
)
from agent_checkpoints.agent_checkpoint_4 import (
    create_agent as create_agent_checkpoint_4,
)
from agent_checkpoints.agent_checkpoint_5 import (
    create_agent as create_agent_checkpoint_5,
)
from agent_checkpoints.agent_checkpoint_6 import (
    create_agent as create_agent_checkpoint_6,
)
from tools import ORDERS_TABLE, SHIPMENTS_TABLE
from dotenv import load_dotenv

list_of_agents = {
    "agent": create_agent_main,
    "agent_checkpoint_1": create_agent_checkpoint_1,
    "agent_checkpoint_2": create_agent_checkpoint_2,
    "agent_checkpoint_3": create_agent_checkpoint_3,
    "agent_checkpoint_4": create_agent_checkpoint_4,
    "agent_checkpoint_5": create_agent_checkpoint_5,
    "agent_checkpoint_6": create_agent_checkpoint_6,
}

load_dotenv()

st.set_page_config(page_title="ReAct Agent Demo", layout="wide")

# --- Make Sidebar Wider ---
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            min-width: 600px;
            max-width: 600px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session State Initialization ---
if "user_id" not in st.session_state:
    st.session_state.user_id = None
    st.session_state.user_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = "agent_checkpoint_1"

if "user_selected_agent" not in st.session_state:
    st.session_state.user_selected_agent = "agent_checkpoint_1"

if (
    "agent" not in st.session_state
    or st.session_state.selected_agent != st.session_state.user_selected_agent
):
    try:
        agent_factory = list_of_agents[st.session_state.user_selected_agent]
        st.session_state.agent = agent_factory()
        st.session_state.selected_agent = st.session_state.user_selected_agent
    except Exception as e:
        st.session_state.agent = None
        st.error(f"Failed to initialize {st.session_state.user_selected_agent}: {e}")


# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    selected_agent = st.selectbox("Select Agent", list(list_of_agents.keys()))
    st.session_state.user_selected_agent = selected_agent

    st.header("User Info")
    # Build mapping of customer_name → customer_id
    customer_map = {
        order.customer_name: getattr(order, "customer_id", None)
        for order in ORDERS_TABLE.values()
    }
    selected_name = st.selectbox("Select User", list(customer_map.keys()))
    st.session_state.user_id = customer_map[selected_name]
    st.session_state.user_name = selected_name

    st.header("⚙️ Settings")

    if st.button("Clear Chat"):
        st.session_state.messages.clear()
        st.rerun()

    st.divider()
    st.subheader("📦 Orders Table")
    orders_df = pd.DataFrame([v.model_dump() for v in ORDERS_TABLE.values()])
    st.dataframe(orders_df, width="stretch")

    st.subheader("🚚 Shipments Table")
    shipments_df = pd.DataFrame([v.model_dump() for v in SHIPMENTS_TABLE.values()])
    st.dataframe(shipments_df, width="stretch")


# --- Chat Window ---
st.title("💬 Customer Support Agent")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "num_iterations" in message:
            st.markdown(f"**Number of llm calls:** {message['num_iterations']}")
        if "actions" in message and message["actions"]:
            with st.expander("🔍 Reasoning trace", expanded=False):
                for step in message["actions"]:
                    if "action" in step:
                        st.markdown(f"**Action:** {step['action']}")
                        if "action_input" in step:
                            st.json(step["action_input"])
                        if "observation" in step:
                            st.markdown(f"**Observation:** {step['observation']}")
                    elif "response" in step:
                        st.markdown(f"**Response:** {step['response']}")
                    st.markdown("---")


# --- Chat Input ---
if prompt := st.chat_input("Ask about orders or shipments..."):
    if not st.session_state.agent:
        st.error(
            "❌ Agent not initialized. Please check your OPENAI_API_KEY environment variable."
        )
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        chat_history = [
            {
                "role": "user",
                "content": f"customer_id: {st.session_state.user_id}, customer_name: {st.session_state.user_name}",
            }
        ]
        # Build chat history for agent
        chat_history += [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
            if m["role"] in ["user", "assistant"]
        ]

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.run(prompt)
                    content = response.get("final_answer", "")
                    reasoning_trace = response.get("reasoning_trace", [])
                    num_iterations = response.get("num_iterations", 0)

                    st.markdown(content)
                    st.markdown(f"**Number of llm calls:** {num_iterations}")
                    if reasoning_trace:
                        with st.expander("🔍 Reasoning trace", expanded=False):
                            for step in reasoning_trace:
                                if "action" in step:
                                    st.markdown(f"**Action:** {step['action']}")
                                    if "action_input" in step:
                                        st.json(step["action_input"])
                                    if "observation" in step:
                                        st.markdown(
                                            f"**Observation:** {step['observation']}"
                                        )
                                elif "response" in step:
                                    st.markdown(f"**Response:** {step['response']}")
                                st.markdown("---")

                    # Add assistant message
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": content,
                            "actions": reasoning_trace,
                            "num_iterations": num_iterations,
                        }
                    )

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"❌ Error: {e}"}
                    )
