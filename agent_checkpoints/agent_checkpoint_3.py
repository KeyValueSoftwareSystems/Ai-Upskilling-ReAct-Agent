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

        # create a tools list
        # bind these tools to the llm

    def _create_system_prompt(self) -> str:
        """Create system prompt with available tools"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        SYSTEM_PROMPT = """
        You are a helpful and professional customer service assistant for an e-commerce company.
        You assist customers by answering questions about their orders, shipments, and related issues.
        Order and Shipment policy:
        - order is shipped within 2 days of order placement
        - shipment is delivered within 5 days of shipping
        - if shipment is not delivered within 5 days, the customer can ask for a refund

        current date: {current_date}

        Strict response format rules (follow exactly):
        never make up your own answer or information, only answer from the order and shipment policy
        if you are clear about the answer, you should ask for confirmation from the customer
        """

        # create a tool description string to be used in the system prompt
        return SYSTEM_PROMPT.format(current_date=current_date)

    def run(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """
        Run the agent on a given query
        """
        self.reasoning_trace = []

        messages = [SystemMessage(content=self._create_system_prompt())]

        messages.append(HumanMessage(content=query))

        try:
            response = self.llm.invoke(messages)

            # check if the llm has responded with a tool call
            # if yes, then execute the tool call
            # call the llm with the tool call response to get the final answer

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
