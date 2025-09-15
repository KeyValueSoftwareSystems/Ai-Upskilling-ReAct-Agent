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
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.messages import HumanMessage, SystemMessage

from tools import (
    get_order,
    get_shipment,
    get_shipment_by_order_id,
    get_order_by_customer_id,
)
from prompt import SYSTEM_PROMPT_TEMPLATE_FOR_CHECKPOINT_6

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
        self.agent = create_react_agent(
            self.llm, self.tools, prompt=self._create_system_prompt()
        )
        self.agent_executor = AgentExecutor(
            agent=self.agent, tools=self.tools, return_intermediate_steps=True
        )
        self.reasoning_trace = []

    def _create_system_prompt(self) -> str:
        """Create system prompt with available tools"""
        return ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE_FOR_CHECKPOINT_6)

    def run(self, query: str, max_iterations: int = 10) -> Dict[str, Any]:
        """
        Run the agent on a given query
        """
        self.reasoning_trace = []

        # add chat history to the messages list to provide context to the llm
        response = self.agent_executor.invoke({"input": query})
        logger.info(f"Response: {response}")

        self.format_intermediate_steps_to_reasoning_trace(response)
        return {
            "final_answer": response["output"],
            "num_iterations": len(self.reasoning_trace),
            "reasoning_trace": self.reasoning_trace,
            "success": True,
        }

    def format_intermediate_steps_to_reasoning_trace(
        self, response: Dict[str, Any]
    ) -> None:

        for item in response.get("intermediate_steps", []):
            # Unpack tuple (AgentAction, observation) or handle unexpected formats
            if isinstance(item, tuple) and len(item) == 2:
                action, observation = item
            else:
                # unknown format — log as a generic observation
                self.reasoning_trace.append(
                    {
                        "iteration": len(self.reasoning_trace) + 1,
                        "response": f"Observation: {json.dumps(item, default=str)}",
                    }
                )
                continue

            # Thought / log (use attribute access, with fallback names)
            thought = getattr(action, "log", None) or getattr(action, "thought", None)
            if thought:
                self.reasoning_trace.append(
                    {
                        "iteration": len(self.reasoning_trace) + 1,
                        "response": f"Thought: {thought}",
                    }
                )

            # Action name and input (use attribute access, with fallbacks)
            action_name = getattr(action, "tool", None) or getattr(
                action, "action", None
            )
            action_input = getattr(action, "tool_input", None) or getattr(
                action, "action_input", None
            )

            self.reasoning_trace.append(
                {
                    "iteration": len(self.reasoning_trace) + 1,
                    "response": f"Action: {action_name}",
                    "action_input": action_input,
                }
            )

            # Observation (pretty-print dicts/lists)
            if isinstance(observation, (dict, list)):
                obs_text = json.dumps(observation, indent=2)
            else:
                obs_text = str(observation)

            self.reasoning_trace.append(
                {
                    "iteration": len(self.reasoning_trace) + 1,
                    "response": f"Observation: {obs_text}",
                }
            )


def create_agent() -> CustomerServiceAgent:
    """
    Create and return a agent instance
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return CustomerServiceAgent(api_key=api_key)  # add api_key here
