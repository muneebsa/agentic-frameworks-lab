# Muneeb Ahmad | https://github.com/muneebsa/agentic-frameworks-lab | MIT License | Educational purposes only

"""
LlamaIndex Weather Assistant - Tool Calling Example
Demonstrates: Function tools, concise API, minimal boilerplate
"""

import asyncio
from dotenv import load_dotenv
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

# Load environment variables
load_dotenv()


# Define tools as simple Python functions
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    weather_data = {
        "San Francisco": "72°F, Sunny",
        "New York": "65°F, Cloudy",
        "Seattle": "58°F, Rainy",
        "Miami": "85°F, Sunny"
    }
    return weather_data.get(city, f"Weather data not available for {city}")


async def main():
    # Initialize LLM
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

    # Create ReAct agent with tools (functions are automatically wrapped)
    agent = ReActAgent(tools=[get_weather], llm=llm, verbose=True)

    # Create context for session state
    ctx = Context(agent)

    # Test queries
    test_queries = [
        "What's the weather in San Francisco?",
        "What's the weather in New York?",
        "Tell me about the weather in Seattle and Miami"
    ]

    print("🌤️  LlamaIndex Weather Assistant\n")

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        # Run agent and await response
        handler = agent.run(query, ctx=ctx)
        response = await handler
        print(f"\nResponse: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())
