# Agentic Frameworks Lab

A comparative study of 6 popular AI agentic frameworks implementing identical use cases side-by-side.

## Frameworks Included

- **AutoGPT** (v0.6.41)
- **CrewAI** (v1.8.1)
- **Google ADK** (v1.22.1)
- **LangChain** (v1.2.6) + LangGraph (v1.0.7)
- **LlamaIndex** (v0.14.13)
- **Microsoft Agent Framework**

## Framework Comparisons

| Comparison | Description |
|----------|-------------|
| **01-llm-tool-calling** | Basic agent with tool execution (weather lookup) |
| **02-multi-agent-orchestration** | Multi-agent travel planning with role specialization |
| **03-rag-implementation** | Product Q&A with FAISS vector database |
| **04-memory-management** | Shopping assistant with persistent memory |

## Quick Start

1. **Clone and setup**
   ```bash
   cd agentic-frameworks-lab
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Configure API keys**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

## Step-by-Step Guide

### Comparison 01: LLM Tool Calling

**Use Case:** Basic agent that can call external tools (weather lookup). Demonstrates fundamental tool execution patterns.

**Run any framework:**
```bash
# AutoGPT
cd framework-comparisons/01-llm-tool-calling/autogpt
pip install -r requirements.txt
python weather_agent.py

# CrewAI
cd framework-comparisons/01-llm-tool-calling/crewai
pip install -r requirements.txt
python weather_agent.py

# Google ADK
cd framework-comparisons/01-llm-tool-calling/google-adk
pip install -r requirements.txt
python weather_agent.py

# LangChain
cd framework-comparisons/01-llm-tool-calling/langchain
pip install -r requirements.txt
python weather_agent.py

# LlamaIndex
cd framework-comparisons/01-llm-tool-calling/llamaindex
pip install -r requirements.txt
python weather_agent.py

# Microsoft Agent Framework
cd framework-comparisons/01-llm-tool-calling/microsoft-agent-framework
pip install -r requirements.txt
python weather_agent.py
```

### Comparison 02: Multi-Agent Orchestration

**Use Case:** Travel planning with 3 specialized agents (Researcher, Booking Specialist, Itinerary Planner) working together. Demonstrates role-based coordination and sequential task execution.

**Run any framework:**
```bash
# AutoGPT
cd framework-comparisons/02-multi-agent-orchestration/autogpt
pip install -r requirements.txt
python travel_planner.py

# CrewAI
cd framework-comparisons/02-multi-agent-orchestration/crewai
pip install -r requirements.txt
python travel_planner.py

# Google ADK
cd framework-comparisons/02-multi-agent-orchestration/google-adk
pip install -r requirements.txt
python travel_planner.py

# LangChain
cd framework-comparisons/02-multi-agent-orchestration/langchain
pip install -r requirements.txt
python travel_planner.py

# LlamaIndex
cd framework-comparisons/02-multi-agent-orchestration/llamaindex
pip install -r requirements.txt
python travel_planner.py

# Microsoft Agent Framework
cd framework-comparisons/02-multi-agent-orchestration/microsoft-agent-framework
pip install -r requirements.txt
python travel_planner.py
```

### Comparison 03: RAG Implementation

**Use Case:** Product Q&A using a FAISS vector database with phone specifications. Demonstrates retrieval-augmented generation for knowledge-based responses.

**Step 1 - Build the shared FAISS index (required once):**
```bash
cd framework-comparisons/03-rag-implementation
pip install -r requirements.txt
python build_index.py
```

**Step 2 - Run any framework:**
```bash
# AutoGPT
cd framework-comparisons/03-rag-implementation/autogpt
pip install -r requirements.txt
python product_qa.py

# CrewAI
cd framework-comparisons/03-rag-implementation/crewai
pip install -r requirements.txt
python product_qa.py

# Google ADK
cd framework-comparisons/03-rag-implementation/google-adk
pip install -r requirements.txt
python product_qa.py

# LangChain
cd framework-comparisons/03-rag-implementation/langchain
pip install -r requirements.txt
python product_qa.py

# LlamaIndex
cd framework-comparisons/03-rag-implementation/llamaindex
pip install -r requirements.txt
python product_qa.py

# Microsoft Agent Framework
cd framework-comparisons/03-rag-implementation/microsoft-agent-framework
pip install -r requirements.txt
python product_qa.py
```

### Comparison 04: Memory Management

**Use Case:** Shopping assistant with persistent memory across conversations. Implements short-term (conversation buffer), long-term (JSON file), and summarization patterns.

**Run any framework:**
```bash
# AutoGPT
cd framework-comparisons/04-memory-management/autogpt
pip install -r requirements.txt
python shopping_assistant.py

# CrewAI
cd framework-comparisons/04-memory-management/crewai
pip install -r requirements.txt
python shopping_assistant.py

# Google ADK
cd framework-comparisons/04-memory-management/google-adk
pip install -r requirements.txt
python shopping_assistant.py

# LangChain
cd framework-comparisons/04-memory-management/langchain
pip install -r requirements.txt
python shopping_assistant.py

# LlamaIndex
cd framework-comparisons/04-memory-management/llamaindex
pip install -r requirements.txt
python shopping_assistant.py

# Microsoft Agent Framework
cd framework-comparisons/04-memory-management/microsoft-agent-framework
pip install -r requirements.txt
python shopping_assistant.py
```

## Project Structure

```
agentic-frameworks-lab/
├── framework-comparisons/
│   ├── 01-llm-tool-calling/
│   ├── 02-multi-agent-orchestration/
│   ├── 03-rag-implementation/
│   └── 04-memory-management/
│       ├── autogpt/
│       ├── crewai/
│       ├── google-adk/
│       ├── langchain/
│       ├── llamaindex/
│       └── microsoft-agent-framework/
├── .env.example
└── README.md
```

## Requirements

- Python 3.x
- OpenAI API key (primary)
- Google Gemini API key (required for Google ADK examples)
