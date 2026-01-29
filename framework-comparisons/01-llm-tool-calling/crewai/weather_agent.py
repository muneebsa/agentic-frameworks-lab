# Muneeb Ahmad | https://github.com/muneebsa/agentic-frameworks-lab | MIT License | Educational purposes only

"""
CrewAI Weather Assistant - Tool Calling Example
Demonstrates: Role-based agents, intuitive API, crew coordination
"""

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# Load environment variables
load_dotenv()

# Configure LLM (BTW, CrewAI implicitly used OpenAI when OPENAI_API_KEY is set)
llm = LLM(model="gpt-3.5-turbo")

# Define tools using @tool decorator
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    weather_data = {
        "San Francisco": "72°F, Sunny",
        "New York": "65°F, Cloudy",
        "Seattle": "58°F, Rainy",
        "Miami": "85°F, Sunny"
    }
    return weather_data.get(city, f"Weather data not available for {city}")


def main():
    # Create weather assistant agent with role and tools
    weather_assistant = Agent(
        role="Weather Assistant",
        goal="Provide accurate weather information",
        backstory="You are an experienced weather assistant who helps people get current weather information for any city.",
        tools=[get_weather],
        llm=llm,
        verbose=True
    )

    # Test queries
    test_queries = [
        "What's the weather in San Francisco?",
        "What's the weather in New York?",
        "Tell me about the weather in Seattle and Miami"
    ]

    print("🌤️  CrewAI Weather Assistant\n")

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        # Create task for this query
        task = Task(
            description=query,
            agent=weather_assistant,
            expected_output="A helpful response with weather information and recommendations"
        )

        # Create crew and execute
        crew = Crew(
            agents=[weather_assistant],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )

        result = crew.kickoff()
        print(f"\nResponse: {result}\n")


if __name__ == "__main__":
    main()
