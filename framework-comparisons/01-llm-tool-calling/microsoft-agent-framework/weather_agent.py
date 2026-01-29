# Muneeb Ahmad | https://github.com/muneebsa/agentic-frameworks-lab | MIT License | Educational purposes only

"""
Microsoft Agent Framework Weather Assistant - Tool Calling Example
Demonstrates: Unified API, function tools with type annotations, async agents
"""

import asyncio
from typing import Annotated
from pydantic import Field
from dotenv import load_dotenv
import os

try:
    from agent_framework import ChatAgent
    from agent_framework.openai import OpenAIChatClient
except ImportError:
    print("Microsoft Agent Framework not installed.")
    print("Install with: pip install agent-framework --pre")
    exit(1)

# Load environment variables
load_dotenv()


# Define tools using type annotations
def get_weather(
    city: Annotated[str, Field(description="The name of the city to get weather for")]
) -> str:
    """Get current weather for a city."""
    weather_data = {
        "San Francisco": "72°F, Sunny",
        "New York": "65°F, Cloudy",
        "Seattle": "58°F, Rainy",
        "Miami": "85°F, Sunny"
    }
    return weather_data.get(city, f"Weather data not available for {city}")


async def main():
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        exit(1)

    # Set API key in environment for OpenAIChatClient
    os.environ["OPENAI_API_KEY"] = api_key

    # Create agent with OpenAI chat client and tools
    agent = ChatAgent(
        chat_client=OpenAIChatClient(model_id="gpt-3.5-turbo"),
        instructions="You are a helpful weather assistant. Use the available tools to answer questions about weather.",
        tools=[get_weather]
    )

    # Test queries
    test_queries = [
        "What's the weather in San Francisco?",
        "What's the weather in New York?",
        "Tell me about the weather in Seattle and Miami"
    ]

    print("🌤️  Microsoft Agent Framework Weather Assistant\n")

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        result = await agent.run(query)
        print(f"\nResponse: {result.text}\n")


if __name__ == "__main__":
    asyncio.run(main())
