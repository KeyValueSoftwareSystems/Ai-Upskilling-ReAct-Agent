# Tic-Tac-Toe AI Agents with LangChain Experimental

This project implements two reasoning agents (Chain-of-Thought and Tree-of-Thought) that can play tic-tac-toe using LangChain Experimental's ToTChain and thought generation strategies.

## 🎯 Features

- **Chain-of-Thought (CoT) Agent**: Uses linear reasoning to generate moves step by step
- **Tree-of-Thought (ToT) Agent**: Explores multiple branches of possible moves before selecting the best one
- **Interactive Streamlit UI**: Play against the agents with real-time reasoning display
- **Real-time Reasoning Traces**: See how each agent thinks through their moves
- **Game State Management**: Complete tic-tac-toe game logic with win detection

## 🏗️ Architecture

### Files Structure

```
├── agent_cot.py          # Chain-of-Thought agent implementation
├── agent_tot.py          # Tree-of-Thought agent implementation
├── tic_tac_toe.py        # Game logic and state management
├── app.py               # Streamlit web application
├── test_agents.py       # Test script for agents
├── requirements.txt     # Python dependencies
└── README_TIC_TAC_TOE.md # This documentation
```

### Agent Implementations

#### CoT Agent (`agent_cot.py`)

- Uses `SampleCoTStrategy` from LangChain Experimental
- Linear reasoning with `c=1` (one thought per step)
- Generates step-by-step reasoning for each move
- Returns move + reasoning trace

#### ToT Agent (`agent_tot.py`)

- Uses `ProposePromptStrategy` from LangChain Experimental
- Multi-branch exploration with `c=3` (three thoughts per step)
- Explores multiple candidate moves before selecting the best
- Returns move + multiple reasoning branches + final decision

### Game Logic (`tic_tac_toe.py`)

- Complete tic-tac-toe game implementation
- Board state management (3x3 grid as 1D array)
- Win condition detection (rows, columns, diagonals)
- Move validation and game status tracking
- Move history and game state serialization

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd Ai-Upskilling-ReAct-Agent
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:

```bash
# Create .env file
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

### Running the Application

1. Start the Streamlit app:

```bash
streamlit run app.py
```

2. Open your browser to `http://localhost:8501`

3. Select an agent type (CoT or ToT) from the sidebar

4. Play tic-tac-toe by clicking on empty squares

5. Watch the agent's reasoning process in real-time

### Testing the Agents

Run the test script to verify agent functionality:

```bash
python test_agents.py
```

## 🎮 How to Play

1. **Select Agent**: Choose between CoT Agent or ToT Agent from the sidebar
2. **Make Move**: Click on an empty square to place your X
3. **Watch Reasoning**: The agent's thinking process is displayed in real-time
4. **Agent Move**: The agent automatically responds with an O
5. **Continue**: Keep playing until someone wins or the board is full
6. **New Game**: Click "New Game" to start over

## 🧠 Agent Reasoning

### Chain-of-Thought (CoT) Agent

- **Strategy**: Linear step-by-step reasoning
- **Process**: Analyzes current state → considers options → selects best move
- **Display**: Shows single reasoning chain with step-by-step thoughts
- **Use Case**: Straightforward, explainable decision making

### Tree-of-Thought (ToT) Agent

- **Strategy**: Multi-branch exploration
- **Process**: Generates multiple candidate moves → evaluates each branch → selects best
- **Display**: Shows multiple reasoning branches + final decision
- **Use Case**: Complex decision making with alternative consideration

## 🔧 Configuration

### Agent Parameters

#### CoT Agent

```python
ToTChain(
    llm=llm,
    c=1,  # Linear reasoning
    thought_generation_strategy=SampleCoTStrategy(llm=llm),
    prompt=custom_prompt
)
```

#### ToT Agent

```python
ToTChain(
    llm=llm,
    c=3,  # Explore 3 branches
    thought_generation_strategy=ProposePromptStrategy(llm=llm),
    prompt=custom_prompt
)
```

### Customization

- **Model**: Change `model_name` in agent constructors (default: "openai/gpt-oss-120b")
- **Branches**: Modify `c` parameter for ToT agent (default: 3)
- **Prompts**: Customize reasoning prompts in `_create_tic_tac_toe_prompt()`

## 🐛 Troubleshooting

### Common Issues

1. **"Agent not initialized" error**

   - Check if `OPENAI_API_KEY` is set in environment
   - Verify API key is valid and has credits

2. **Import errors**

   - Ensure `langchain-experimental` is installed
   - Check Python version compatibility

3. **Agent makes invalid moves**
   - Check game state formatting
   - Verify move parsing logic

### Debug Mode

Enable verbose logging by setting:

```python
strategy = SampleCoTStrategy(llm=self.llm, verbose=True)
```

## 📊 Performance

### CoT Agent

- **Speed**: Faster (single reasoning path)
- **Memory**: Lower (linear trace)
- **Explainability**: High (clear step-by-step)

### ToT Agent

- **Speed**: Slower (multiple reasoning paths)
- **Memory**: Higher (multiple branches)
- **Explainability**: Very High (shows alternatives)

## 🔮 Future Enhancements

- [ ] Add difficulty levels
- [ ] Implement different game modes (AI vs AI)
- [ ] Add move evaluation scores
- [ ] Support for different board sizes
- [ ] Tournament mode with multiple agents
- [ ] Performance metrics and analytics

## 📝 License

This project is part of the AI Upskilling ReAct Agent demonstration.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions or issues, please check the troubleshooting section or create an issue in the repository.
