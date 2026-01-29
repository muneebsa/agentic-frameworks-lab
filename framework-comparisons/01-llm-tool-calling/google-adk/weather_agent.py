# Muneeb Ahmad | https://github.com/muneebsa/agentic-frameworks-lab | MIT License | Educational purposes only

"""
Google ADK Weather Assistant - Tool Calling Example
Demonstrates: Code-first approach, automatic function tools, Gemini integration
"""

import asyncio
import os
import warnings
from dotenv import load_dotenv

# Suppress Google ADK warnings for cleaner educational output
# These warnings are about experimental features and function_call parts in responses
warnings.filterwarnings('ignore', category=UserWarning, module='google.adk')
warnings.filterwarnings('ignore', message='.*non-text parts in the response.*')

try:
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
except ImportError:
    print("Google ADK not installed.")
    print("Install with: pip install google-adk")
    exit(1)

# Load environment variables
load_dotenv()


# Define tool as a simple Python function
def get_weather(city: str) -> dict:
    """
    Get current weather for a city.

    Args:
        city: The name of the city to get weather for.

    Returns:
        Dictionary with weather information.
    """
    weather_data = {
        "San Francisco": "72°F, Sunny",
        "New York": "65°F, Cloudy",
        "Seattle": "58°F, Rainy",
        "Miami": "85°F, Sunny"
    }

    result = weather_data.get(city, f"Weather data not available for {city}")
    return {"status": "success", "city": city, "weather": result}


async def main():
    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment")
        print("Get your API key from: https://aistudio.google.com/apikey")
        exit(1)

    # Set API key in environment
    os.environ["GOOGLE_API_KEY"] = api_key

    # Create agent with tools (function is automatically wrapped as FunctionTool)
    weather_agent = Agent(
        model='gemini-2.5-flash',
        name='weather_agent',
        description='A helpful weather assistant',
        instruction='You are a helpful weather assistant. Use the get_weather tool to answer questions about weather in different cities.',
        tools=[get_weather]
    )

    # Setup session service
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="weather_app",
        user_id="user123"
    )

    # Create runner
    runner = Runner(
        agent=weather_agent,
        app_name="weather_app",
        session_service=session_service
    )

    # Test queries
    test_queries = [
        "What's the weather in San Francisco?",
        "What's the weather in New York?",
        "Tell me about the weather in Seattle and Miami"
    ]

    print("🌤️  Google ADK Weather Assistant\n")

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        # Create content message
        content = types.Content(
            role='user',
            parts=[types.Part(text=query)]
        )

        # Run agent and collect response
        response_text = ""
        async for event in runner.run_async(
            user_id="user123",
            session_id=session.id,
            new_message=content
        ):
            if event.is_final_response():
                response_text = event.content.parts[0].text

        print(f"\nResponse: {response_text}\n")


if __name__ == "__main__":
    asyncio.run(main())
