# Muneeb Ahmad | https://github.com/muneebsa/agentic-frameworks-lab | MIT License | Educational purposes only

"""
LangChain/LangGraph Weather Assistant - Tool Calling Example
Demonstrates: Tool definition with decorators, graph-based agent, minimal code
"""

# Suppress transformers deprecation warning BEFORE imports (warning occurs during import)
import warnings
warnings.filterwarnings('ignore', message='.*torch.utils._pytree.*')

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

# Load environment variables
load_dotenv()

# Define tools using @tool decorator
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Mock implementation - would call real weather API in production
    weather_data = {
        "San Francisco": "72°F, Sunny",
        "New York": "65°F, Cloudy",
        "Seattle": "58°F, Rainy",
        "Miami": "85°F, Sunny"
    }
    return weather_data.get(city, f"Weather data not available for {city}")


def main():
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Combine tools
    tools = [get_weather]

    # Create agent using LangGraph (modern approach, replaces AgentExecutor)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a helpful weather assistant. Use the get_weather tool to answer questions about weather."
    )

    # Test queries
    test_queries = [
        "What's the weather in San Francisco?",
        "What's the weather in New York?",
        "Tell me about the weather in Seattle and Miami"
    ]

    print("🌤️  LangChain/LangGraph Weather Assistant\n")

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        # Invoke agent with messages format
        result = agent.invoke({"messages": [("human", query)]})

        # Extract the final response from messages
        final_message = result["messages"][-1]
        print(f"\nResponse: {final_message.content}\n")


if __name__ == "__main__":
    main()
