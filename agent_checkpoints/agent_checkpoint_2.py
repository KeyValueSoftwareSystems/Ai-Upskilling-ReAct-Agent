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

from langchain_core.messages import HumanMessage, SystemMessage
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
        self.llm = ChatOpenAI(api_key=api_key, model_name=model_name)
        self.reasoning_trace = []

    # create a function to create a system prompt

    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the agent on a given query
        """
        self.reasoning_trace = []

        # Create conversation history with system prompt
        messages = []  # add system prompt in the messages list

        # Add current query
        messages.append(HumanMessage(content=query))

        try:
            response = self.llm.invoke(messages)
            logger.info(f"Response: {response.content}")
            return {
                "final_answer": response.content,
                "reasoning_trace": self.reasoning_trace,
                "num_iterations": 1,
                "success": True,
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
    return CustomerServiceAgent(api_key=api_key)
