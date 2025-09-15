import json
import os
from datetime import datetime
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from loguru import logger

from tools import (
    get_order,
    get_shipment,
    get_shipment_by_order_id,
    get_order_by_customer_id,
)


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

    def _create_system_prompt(self) -> str:
        """Create system prompt with available tools"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        SYSTEM_PROMPT = """
        You are a helpful and professional customer service assistant for an e-commerce company.
        You assist customers by answering questions about their orders, shipments, and related issues.
        
        Order and shipment policy:
        - Order is shipped within 2 days of order placement
        - Shipment is delivered within 5 days of shipping
        - If shipment is not delivered within 5 days, the customer can ask for a refund

        Current date: {current_date}

        You have access to the following tools:
        {tools}

        IMPORTANT:
        - If you need external data, you must call the appropriate tool.
        - If you are clear about the answer, you may give a final answer directly.
        - Do not guess data you could get from a tool.
        """
        tool_descs = "\n".join([f"- {t.name}: {t.description}" for t in self.tools])
        return SYSTEM_PROMPT.format(current_date=current_date, tools=tool_descs)

    def run(self, query: str, max_iterations: int = 5) -> Dict[str, Any]:
        """
        Run the agent in a ReAct loop: LLM → Tool → LLM → Final answer.
        """
        self.reasoning_trace = []
        messages = [SystemMessage(content=self._create_system_prompt())]
        messages.append(HumanMessage(content=query))

        for i in range(1, max_iterations + 1):
            logger.info(f"--- Iteration {i} ---")
            response = self.llm_with_tools.invoke(messages)
            logger.info(f"Response: {response}")

            # If LLM wants to call a tool
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                tool_name = tool_call["name"]
                tool_input = tool_call["args"]
                tool_id = tool_call["id"]

                logger.info(f"Tool requested: {tool_name} with args {tool_input}")

                if tool_name not in self.tools_by_name:
                    return {
                        "final_answer": f"Error: Unknown tool {tool_name}",
                        "reasoning_trace": self.reasoning_trace,
                        "num_iterations": i,
                        "success": False,
                    }

                # Run the tool
                tool = self.tools_by_name[tool_name]
                try:
                    tool_response = tool.invoke(tool_input)
                except Exception as e:
                    tool_response = {"error": str(e)}
                logger.info(f"Tool response: {tool_response}")

                # Record reasoning trace
                self.reasoning_trace.append(
                    {
                        "iteration": i,
                        "action": tool_name,
                        "action_input": tool_input,
                        "observation": tool_response,
                    }
                )

                # Append tool call + tool result back to messages
                messages.append(response)  # the AIMessage with tool_call
                messages.append(
                    ToolMessage(
                        content=json.dumps(tool_response),
                        tool_call_id=tool_id,
                    )
                )
                continue  # go to next iteration

            # Otherwise → Final answer
            self.reasoning_trace.append({"iteration": i, "response": response.content})
            return {
                "final_answer": response.content,
                "reasoning_trace": self.reasoning_trace,
                "num_iterations": i,
                "success": True,
            }

        # Fallback: reached max iterations
        return {
            "final_answer": "Reached max iterations without final answer.",
            "reasoning_trace": self.reasoning_trace,
            "num_iterations": max_iterations,
            "success": False,
        }


def create_agent() -> CustomerServiceAgent:
    """Factory function to create the agent"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return CustomerServiceAgent(api_key=api_key)
