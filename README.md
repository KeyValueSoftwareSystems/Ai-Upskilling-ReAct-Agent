# ReAct Agent Learning Project

A hands-on project to learn how to build a ReAct (Reasoning and Acting) agent. This project will guide you through creating an intelligent agent that can reason through problems and take actions step by step.

## Prerequisites

Before you begin, make sure you have:

- **Python 3.11** installed on your system
- An **OpenAI API key** (we'll set this up in the setup guide below)

## Quick Setup Guide

Follow these simple steps to get started:

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd Ai-Upskilling-ReAct-Agent
   ```

2. **Create and activate a virtual environment:**

   **Option A: Using conda (if you have conda installed)**

   ```bash
   # Create conda environment
   conda create -n venv python=3.11

   # Activate conda environment
   conda activate venv
   ```

   **Option B: Using venv (if you don't have conda)**

   ```bash
   # Create virtual environment
   python3.11 -m venv venv

   # Activate virtual environment
   # On Linux/Mac:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API key:**

   Create a `.env` file in the project root:

   ```bash
   # Create .env file
   touch .env
   ```

   Add your OpenAI API key to the `.env` file:

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   **To get an OpenAI API key:**

   - Go to [OpenAI Platform](https://platform.openai.com/api-keys)
   - Sign up or log in to your account
   - Click "Create new secret key"
   - Copy the key and paste it in your `.env` file

5. **Run the application:**

   ```bash
   streamlit run app.py
   ```

6. **Open your browser:**
   - Go to `http://localhost:8501`
   - Select "agent" from the dropdown menu in the left sidebar
   - Start building your AI agent!

## Prerequisites Confirmation

After running the application, you should see an interface similar to this:

The interface should show:

- A left sidebar with settings and data tables (Orders and Shipments)
- A chat interface on the right with sample conversation
- Dropdown menus for agent and user selection
- Pre-populated data in the tables

## What You'll Learn

This project will teach you:

- How AI agents think and make decisions
- How to build tools that agents can use
- How to create interactive web interfaces for your agents
- How to handle different types of user questions

## Sample Queries

Once your agent is set up, you can ask questions like:

- "Where is my order #1234?"
- "What's the status of order #1235?"
- "Tell me about shipment #1001"
- "Is order #1236 delivered?"
- "Show me all orders for customer 1001"
- "What orders did customer 1001 place in September 2025?"

## How It Works

1. **User Input**: User enters a question in the Streamlit interface
2. **Agent Reasoning**: The ReAct agent analyzes the question and decides what action to take
3. **Tool Execution**: Agent calls appropriate tools (get_order, get_shipment, etc.)
4. **Observation**: Agent processes the tool results
5. **Iteration**: Agent continues reasoning until it has enough information
6. **Final Answer**: Agent provides a complete, grounded answer

## Architecture

- **tools.py**: Defines Pydantic schemas and tool functions
- **agent.py**: Implements the ReAct agent with LangChain integration
- **app.py**: Provides the Streamlit web interface

## Project Structure

```
├── app.py          # Web interface for your agent
├── agent.py        # Your AI agent (starts with basic code)
├── tools.py        # Tools your agent can use
├── requirements.txt # Python packages needed
└── README.md       # This guide
```

## Getting Started

The project starts with basic code that you'll build upon. As you progress through the learning modules, you'll:

- Add more capabilities to your agent
- Create new tools for different tasks
- Improve how your agent responds to users
- Learn best practices for AI agent development

## Need Help?

If you run into any issues:

1. Make sure you're using Python 3.11
2. Check that your virtual environment is activated
3. Verify your `.env` file has the correct API key
4. Ensure all dependencies are installed with `pip install -r requirements.txt`
