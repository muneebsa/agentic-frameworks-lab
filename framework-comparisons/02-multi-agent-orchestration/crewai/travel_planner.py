# Muneeb Ahmad | https://github.com/muneebsa/agentic-frameworks-lab | MIT License | Educational purposes only

"""
CrewAI Travel Planner - Multi-Agent Orchestration Example
Demonstrates: Crew workflow with Process.sequential, role-based agents, task coordination
"""

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# Load environment variables
load_dotenv()

# Configure LLM
llm = LLM(model="gpt-3.5-turbo", temperature=0.7)

# Define tools
@tool
def search_destinations(destination: str) -> str:
    """Search for travel information about a destination."""
    # Mock implementation
    destinations = {
        "Paris": "Paris offers the Eiffel Tower, Louvre Museum, and Notre-Dame. Best time: April-June. Average cost: $200/day.",
        "Tokyo": "Tokyo features temples, cherry blossoms, and modern tech. Best time: March-May. Average cost: $180/day.",
        "Bali": "Bali has beaches, temples, and rice terraces. Best time: April-October. Average cost: $100/day."
    }
    return destinations.get(destination, f"Information about {destination}: Beautiful destination with rich culture.")

@tool
def check_availability(destination: str, dates: str) -> str:
    """Check hotel and flight availability."""
    # Mock implementation
    return f"Available for {destination} during {dates}: Hotels from $100/night, Flights from $500 roundtrip."

def main():
    # Create specialized agents
    researcher = Agent(
        role="Travel Researcher",
        goal="Research destinations and provide detailed travel information",
        backstory="You are an experienced travel researcher who knows the best places to visit and optimal travel times.",
        tools=[search_destinations],
        llm=llm,
        verbose=True
    )

    booking_agent = Agent(
        role="Booking Specialist",
        goal="Find and recommend the best hotels and flights",
        backstory="You are a booking specialist with access to the best deals on accommodations and transportation.",
        tools=[check_availability],
        llm=llm,
        verbose=True
    )

    itinerary_planner = Agent(
        role="Itinerary Planner",
        goal="Create detailed day-by-day travel itineraries",
        backstory="You are an expert at creating well-structured, enjoyable travel itineraries that maximize the experience.",
        llm=llm,
        verbose=True
    )

    # Define travel planning input
    destination = "Paris"
    dates = "June 15-22, 2026"
    preferences = "art, history, local cuisine"

    print(f"🌍 CrewAI Travel Planner\n")
    print(f"Planning trip to: {destination}")
    print(f"Dates: {dates}")
    print(f"Interests: {preferences}\n")
    print("="*60)

    # Create tasks
    research_task = Task(
        description=f"Research {destination} and provide key attractions, best time to visit, and estimated costs. Focus on {preferences}.",
        agent=researcher,
        expected_output="Detailed research report with attractions and recommendations"
    )

    booking_task = Task(
        description=f"Check availability and recommend hotels and flights for {destination} during {dates}.",
        agent=booking_agent,
        expected_output="Hotel and flight recommendations with pricing"
    )

    itinerary_task = Task(
        description=f"Create a detailed 7-day itinerary for {destination} incorporating the research findings and booking recommendations. Focus on {preferences}.",
        agent=itinerary_planner,
        expected_output="Complete day-by-day itinerary with activities and logistics"
    )

    # Create crew with sequential process
    crew = Crew(
        agents=[researcher, booking_agent, itinerary_planner],
        tasks=[research_task, booking_task, itinerary_task],
        process=Process.sequential,
        verbose=True
    )

    # Execute the crew
    result = crew.kickoff()

    print("\n" + "="*60)
    print("📋 Final Travel Plan:")
    print("="*60)
    print(result)


if __name__ == "__main__":
    main()
