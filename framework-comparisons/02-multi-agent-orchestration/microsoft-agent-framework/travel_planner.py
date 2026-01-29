# Muneeb Ahmad | https://github.com/muneebsa/agentic-frameworks-lab | MIT License | Educational purposes only

"""
Microsoft Agent Framework Travel Planner - Multi-Agent Orchestration Example
Demonstrates: Sequential workflow orchestration, agent pipeline, shared conversation context
"""

import asyncio
from typing import Annotated
from pydantic import Field
from dotenv import load_dotenv
import os

try:
    from agent_framework import SequentialBuilder, WorkflowOutputEvent, ChatMessage
    from agent_framework.openai import OpenAIChatClient
except ImportError:
    print("Microsoft Agent Framework not installed.")
    print("Install with: pip install agent-framework --pre")
    exit(1)

# Load environment variables
load_dotenv()

# Define tools
def search_destinations(
    destination: Annotated[str, Field(description="The destination to research")]
) -> str:
    """Search for travel information about a destination."""
    destinations = {
        "Paris": "Paris offers the Eiffel Tower, Louvre Museum, and Notre-Dame. Best time: April-June. Average cost: $200/day.",
        "Tokyo": "Tokyo features temples, cherry blossoms, and modern tech. Best time: March-May. Average cost: $180/day.",
        "Bali": "Bali has beaches, temples, and rice terraces. Best time: April-October. Average cost: $100/day."
    }
    return destinations.get(destination, f"Information about {destination}: Beautiful destination with rich culture.")

def check_availability(
    destination: Annotated[str, Field(description="The destination")],
    dates: Annotated[str, Field(description="Travel dates")]
) -> str:
    """Check hotel and flight availability."""
    return f"Available for {destination} during {dates}: Hotels from $100/night, Flights from $500 roundtrip."

async def main():
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        exit(1)

    os.environ["OPENAI_API_KEY"] = api_key

    # Create chat client
    chat_client = OpenAIChatClient(model_id="gpt-3.5-turbo")

    # Create specialized agents for sequential workflow
    researcher = chat_client.as_agent(
        instructions="You are a travel researcher. Research the destination and provide key attractions and costs.",
        name="researcher",
        tools=[search_destinations]
    )

    booking_agent = chat_client.as_agent(
        instructions="You are a booking specialist. Review the research and check availability for hotels and flights.",
        name="booking",
        tools=[check_availability]
    )

    itinerary_planner = chat_client.as_agent(
        instructions="You are an itinerary planner. Create a detailed 7-day travel plan based on the research and booking information.",
        name="itinerary"
    )

    # Build sequential workflow: researcher -> booking_agent -> itinerary_planner
    workflow = SequentialBuilder().participants([
        researcher,
        booking_agent,
        itinerary_planner
    ]).build()

    # Define travel planning input
    destination = "Paris"
    dates = "June 15-22, 2026"
    preferences = "art, history, local cuisine"

    print(f"🌍 Microsoft Agent Framework Travel Planner\n")
    print(f"Planning trip to: {destination}")
    print(f"Dates: {dates}")
    print(f"Interests: {preferences}\n")
    print("="*60)

    # Execute the sequential workflow
    print("\n🔄 Executing sequential workflow...")
    output_evt: WorkflowOutputEvent | None = None
    async for event in workflow.run_stream(
        f"Plan a trip to {destination} from {dates}, focusing on {preferences}."
    ):
        if isinstance(event, WorkflowOutputEvent):
            output_evt = event

    # Display final result
    if output_evt:
        print("\n" + "="*60)
        print("📋 Final Travel Plan:")
        print("="*60)
        messages: list[ChatMessage] = output_evt.data
        # Get the last message from the itinerary planner
        for msg in reversed(messages):
            if msg.author_name == "itinerary":
                print(msg.text)
                break


if __name__ == "__main__":
    asyncio.run(main())
