# Muneeb Ahmad | https://github.com/muneebsa/agentic-frameworks-lab | MIT License | Educational purposes only

"""
Google ADK Travel Planner - Multi-Agent Orchestration Example
Demonstrates: SequentialAgent workflow, LlmAgent coordination, state sharing
"""

import asyncio
import os
import warnings
from dotenv import load_dotenv

# Suppress Google ADK warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.adk')
warnings.filterwarnings('ignore', message='.*non-text parts in the response.*')

try:
    from google.adk.agents import LlmAgent, SequentialAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
except ImportError:
    print("Google ADK not installed.")
    print("Install with: pip install google-adk")
    exit(1)

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
    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment")
        exit(1)

    os.environ["GOOGLE_API_KEY"] = api_key

    # Define travel planning input
    destination = "Paris"
    dates = "June 15-22, 2026"
    preferences = "art, history, local cuisine"

    print(f"🌍 Google ADK Travel Planner\n")
    print(f"Planning trip to: {destination}")
    print(f"Dates: {dates}")
    print(f"Interests: {preferences}\n")
    print("="*60)

    # Create specialized LlmAgents with output_key for state sharing
    researcher = LlmAgent(
        name="ResearchAgent",
        model="gemini-2.5-flash",
        instruction=f"Research {destination} focusing on {preferences}. Provide key attractions and costs.",
        tools=[search_destinations],
        output_key="research_output"
    )

    booking_agent = LlmAgent(
        name="BookingAgent",
        model="gemini-2.5-flash",
        instruction=f"Check availability for {destination} during {dates}. Based on research: {{research_output}}",
        tools=[check_availability],
        output_key="booking_output"
    )

    itinerary_planner = LlmAgent(
        name="ItineraryAgent",
        model="gemini-2.5-flash",
        instruction=f"Create a 7-day itinerary for {destination} focusing on {preferences}. Use research: {{research_output}} and booking: {{booking_output}}",
        output_key="final_itinerary"
    )

    # Create SequentialAgent to orchestrate the workflow
    root_agent = SequentialAgent(
        name="TravelPlannerWorkflow",
        sub_agents=[researcher, booking_agent, itinerary_planner],
        description="Executes travel planning: research, booking, and itinerary creation"
    )

    # Setup session service and runner
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="travel_planner",
        user_id="user123"
    )

    runner = Runner(
        agent=root_agent,
        app_name="travel_planner",
        session_service=session_service
    )

    # Execute the sequential workflow
    print("\n🔄 Executing sequential workflow...")
    message_content = types.Content(
        role='user',
        parts=[types.Part(text=f"Plan a trip to {destination}")]
    )

    final_output = ""
    async for event in runner.run_async(
        user_id="user123",
        session_id=session.id,
        new_message=message_content
    ):
        if event.is_final_response() and event.content:
            final_output = event.content.parts[0].text if event.content.parts else ""

    print("\n" + "="*60)
    print("📋 Final Travel Plan:")
    print("="*60)
    print(final_output)


if __name__ == "__main__":
    asyncio.run(main())
