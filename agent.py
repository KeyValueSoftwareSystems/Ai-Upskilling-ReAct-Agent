"""
agent.py - ReAct agent implementation using LangChain and Gemini

This file contains:
- ReAct agent class that implements reasoning loops
- Integration with Gemini LLM via LangChain
- Tool calling and structured output handling
- Reasoning trace tracking for debugging
"""

import json
from datetime import datetime
import os
import re
from typing import Dict, Any, Optional, Tuple
from langchain_openai import ChatOpenAI

from langchain_core.messages import HumanMessage
from tools import (
    get_order,
    get_shipment,
    get_shipment_by_order_id,
    get_order_by_customer_id,
)
from prompt import SYSTEM_PROMPT_TEMPLATE

from loguru import logger


class CustomerServiceAgent:

    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        # use ChatOpenAI to create a llm client
        pass

    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the agent on a given query
        """
        self.reasoning_trace = []

        # Create conversation history with system prompt
        messages = []  # add system prompt here

        # Add current query
        messages.append(HumanMessage(content=query))

        try:
            # use the initialized llm client to invoke the messages
            return {
                "final_answer": "Hi, I'm a customer service agent. How can I help you today?",
                "reasoning_trace": self.reasoning_trace,
                "success": False,
            }

        except Exception as e:
            error_msg = f"Error in iteration {1}: {str(e)}"
            self.reasoning_trace.append({"iteration": 1, "error": error_msg})
            return {
                "final_answer": f"I encountered an error: {error_msg}",
                "reasoning_trace": self.reasoning_trace,
                "num_iterations": 1,
                "success": False,
            }


def create_agent() -> CustomerServiceAgent:
    """
    Create and return a agent instance
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return CustomerServiceAgent(api_key=api_key)  # add api_key here
