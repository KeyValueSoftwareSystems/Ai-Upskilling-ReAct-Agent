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

from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage, AIMessage

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
        self.tools = [
            get_order,
            get_shipment,
            get_shipment_by_order_id,
            get_order_by_customer_id,
        ]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.reasoning_trace = []

    # modify the system prompt to instruct the llm to reason before creating a response

    def _create_system_prompt(self) -> str:
        """Create system prompt with available tools"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        SYSTEM_PROMPT = """
        You are a helpful and professional customer service assistant for an e-commerce company.
        You assist customers by answering questions about their orders, shipments, and related issues.
        order and shipment policy:
        - order is shipped within 2 days of order placement
        - shipment is delivered within 5 days of shipping
        - if shipment is not delivered within 5 days, the customer can ask for a refund

        current date: {current_date}

        you have access to the following tools:
        {tools}

        For responding, you can either call a tool or provide a final answer.
        if you are clear about the answer, you should ask for confirmation from the customer
        """
        tool_descs = "\n".join([f"- {t.name}: {t.description}" for t in self.tools])
        return SYSTEM_PROMPT.format(current_date=current_date, tools=tool_descs)

    # add a function to extract the tool call from the llm response
    # tool call input parameters will be present in the llm response
    # add a function to execute the tool call

    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the agent on a given query
        """
        self.reasoning_trace = []

        messages = [SystemMessage(content=self._create_system_prompt())]

        messages.append(HumanMessage(content=query))

        try:
            response = self.llm_with_tools.invoke(messages)
            logger.info(f"Response: {response}")

            # we have to provide the llm multiple iterations
            # so that agent can continue its execution flow
            # reason -> action -> observe or reason -> final answer
            # here we will call the functions to execute the tool call

            if response.tool_calls:
                tool_call = response.tool_calls[0]
                tool_name = tool_call["name"]
                tool_input = tool_call["args"]
                tool_id = tool_call["id"]
                tool = self.tools_by_name[tool_name]

                messages.append(response)
                logger.info(f"Tool input: {tool_input}")
                tool_response = tool.invoke(tool_input)
                logger.info(f"Tool response: {tool_response}")

                messages.append(
                    ToolMessage(
                        content=json.dumps(tool_response),
                        tool_call_id=tool_id,
                    )
                )
                self.reasoning_trace.append(
                    {
                        "iteration": 1,
                        "action": tool_name,
                        "action_input": tool_input,
                        "observation": tool_response,
                    }
                )

                response = self.llm.invoke(messages)
                logger.info(f"Response: {response}")
                self.reasoning_trace.append(
                    {"iteration": 2, "response": response.content}
                )
                return {
                    "final_answer": response.content,
                    "reasoning_trace": self.reasoning_trace,
                    "num_iterations": 2,
                    "success": True,
                }

            else:
                self.reasoning_trace.append(
                    {"iteration": 1, "response": response.content}
                )
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
