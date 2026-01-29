# Muneeb Ahmad | https://github.com/muneebsa/agentic-frameworks-lab | MIT License | Educational purposes only

"""
LlamaIndex Travel Planner - Multi-Agent Orchestration Example
Demonstrates: Multi-agent sequential workflow with FunctionAgent, context management
"""

import asyncio
from dotenv import load_dotenv
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

# Load environment variables
load_dotenv()

# Define tools
def search_destinations(destination: str) -> str:
    """Search for travel information about a destination."""
    destinations = {
        "Paris": "Paris offers the Eiffel Tower, Louvre Museum, and Notre-Dame. Best time: April-June. Average cost: $200/day.",
        "Tokyo": "Tokyo features temples, cherry blossoms, and modern tech. Best time: March-May. Average cost: $180/day.",
        "Bali": "Bali has beaches, temples, and rice terraces. Best time: April-October. Average cost: $100/day."
    }
    return destinations.get(destination, f"Information about {destination}: Beautiful destination with rich culture.")

def check_availability(destination: str, dates: str) -> str:
    """Check hotel and flight availability."""
    return f"Available for {destination} during {dates}: Hotels from $100/night, Flights from $500 roundtrip."

async def main():
    # Initialize LLM
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # Create specialized FunctionAgents
    research_agent = FunctionAgent(
        name="ResearchAgent",
        description="Researches destinations",
        system_prompt="You are a travel researcher. Provide detailed information about destinations.",
        llm=llm,
        tools=[search_destinations]
    )

    booking_agent = FunctionAgent(
        name="BookingAgent",
        description="Checks availability for hotels and flights",
        system_prompt="You are a booking specialist. Check availability and provide recommendations.",
        llm=llm,
        tools=[check_availability]
    )

    itinerary_agent = FunctionAgent(
        name="ItineraryAgent",
        description="Creates detailed travel itineraries",
        system_prompt="You are an itinerary planner. Create detailed day-by-day travel plans.",
        llm=llm,
        tools=[]
    )

    # Define travel planning input
    destination = "Paris"
    dates = "June 15-22, 2026"
    preferences = "art, history, local cuisine"

    print(f"🌍 LlamaIndex Travel Planner\n")
    print(f"Planning trip to: {destination}")
    print(f"Dates: {dates}")
    print(f"Interests: {preferences}\n")
    print("="*60)

    # Execute agents sequentially with context
    print("\n🔍 Step 1: Research Agent")
    research_ctx = Context(research_agent)
    research_response = await research_agent.run(
        user_msg=f"Research {destination} focusing on {preferences}. Provide key attractions and costs.",
        ctx=research_ctx
    )
    research_output = str(research_response)

    print("\n✈️  Step 2: Booking Agent")
    booking_ctx = Context(booking_agent)
    booking_response = await booking_agent.run(
        user_msg=f"Check availability for {destination} during {dates}.",
        ctx=booking_ctx
    )
    booking_output = str(booking_response)

    print("\n📅 Step 3: Itinerary Agent")
    itinerary_ctx = Context(itinerary_agent)
    itinerary_response = await itinerary_agent.run(
        user_msg=f"Create a 7-day itinerary for {destination} from {dates} based on:\n\nResearch: {research_output}\n\nBooking: {booking_output}\n\nFocus on {preferences}.",
        ctx=itinerary_ctx
    )

    print("\n" + "="*60)
    print("📋 Final Travel Plan:")
    print("="*60)
    print(str(itinerary_response))


if __name__ == "__main__":
    asyncio.run(main())
