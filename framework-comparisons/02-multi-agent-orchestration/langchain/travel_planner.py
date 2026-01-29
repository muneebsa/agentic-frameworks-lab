# Muneeb Ahmad | https://github.com/muneebsa/agentic-frameworks-lab | MIT License | Educational purposes only

"""
LangChain/LangGraph Travel Planner - Multi-Agent Orchestration Example
Demonstrates: StateGraph workflow, multi-agent coordination, shared state management
"""

# Suppress warnings before imports
import warnings
warnings.filterwarnings('ignore', message='.*torch.utils._pytree.*')

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Load environment variables
load_dotenv()

# Define tools
@tool
def search_destinations(destination: str) -> str:
    """Search for travel information about a destination."""
    destinations = {
        "Paris": "Paris offers the Eiffel Tower, Louvre Museum, and Notre-Dame. Best time: April-June. Average cost: $200/day.",
        "Tokyo": "Tokyo features temples, cherry blossoms, and modern tech. Best time: March-May. Average cost: $180/day.",
        "Bali": "Bali has beaches, temples, and rice terraces. Best time: April-October. Average cost: $100/day."
    }
    return destinations.get(destination, f"Information about {destination}: Beautiful destination with rich culture.")

@tool
def check_availability(destination: str, dates: str) -> str:
    """Check hotel and flight availability."""
    return f"Available for {destination} during {dates}: Hotels from $100/night, Flights from $500 roundtrip."

# Define state for workflow
class TravelPlanState(TypedDict):
    destination: str
    dates: str
    preferences: str
    research_output: str
    booking_output: str
    final_plan: str

def main():
    # Initialize LLM and agents
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    researcher = create_agent(
        model=llm,
        tools=[search_destinations],
        system_prompt="You are a travel researcher. Research destinations and provide detailed information."
    )

    booking_agent = create_agent(
        model=llm,
        tools=[check_availability],
        system_prompt="You are a booking specialist. Check availability and recommend hotels and flights."
    )

    itinerary_agent = create_agent(
        model=llm,
        tools=[],
        system_prompt="You are an itinerary planner. Create detailed day-by-day travel plans."
    )

    # Define node functions that update state
    def research_node(state: TravelPlanState) -> TravelPlanState:
        result = researcher.invoke({
            "messages": [("human", f"Research {state['destination']} focusing on {state['preferences']}.")]
        })
        state["research_output"] = result["messages"][-1].content
        return state

    def booking_node(state: TravelPlanState) -> TravelPlanState:
        result = booking_agent.invoke({
            "messages": [("human", f"Check availability for {state['destination']} during {state['dates']}.")]
        })
        state["booking_output"] = result["messages"][-1].content
        return state

    def itinerary_node(state: TravelPlanState) -> TravelPlanState:
        result = itinerary_agent.invoke({
            "messages": [("human", f"Create a 7-day itinerary for {state['destination']} based on:\nResearch: {state['research_output']}\nBooking: {state['booking_output']}\nPreferences: {state['preferences']}")]
        })
        state["final_plan"] = result["messages"][-1].content
        return state

    # Build StateGraph workflow
    workflow = StateGraph(TravelPlanState)

    # Add nodes for each agent
    workflow.add_node("researcher", research_node)
    workflow.add_node("booking", booking_node)
    workflow.add_node("itinerary", itinerary_node)

    # Define the workflow edges (sequential flow)
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "booking")
    workflow.add_edge("booking", "itinerary")
    workflow.add_edge("itinerary", END)

    # Compile the workflow
    app = workflow.compile()

    # Define input
    destination = "Paris"
    dates = "June 15-22, 2026"
    preferences = "art, history, local cuisine"

    print(f"🌍 LangChain/LangGraph Travel Planner\n")
    print(f"Planning trip to: {destination}")
    print(f"Dates: {dates}")
    print(f"Interests: {preferences}\n")
    print("="*60)

    # Execute workflow
    print("\n🔄 Executing StateGraph workflow...")
    result = app.invoke({
        "destination": destination,
        "dates": dates,
        "preferences": preferences,
        "research_output": "",
        "booking_output": "",
        "final_plan": ""
    })

    print("\n" + "="*60)
    print("📋 Final Travel Plan:")
    print("="*60)
    print(result["final_plan"])


if __name__ == "__main__":
    main()
