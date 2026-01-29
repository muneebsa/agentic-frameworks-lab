# Muneeb Ahmad | https://github.com/muneebsa/agentic-frameworks-lab | MIT License | Educational purposes only

"""
AutoGPT Travel Planner - Multi-Agent Orchestration Example
Demonstrates: Command-based multi-agent system, autonomous coordination, function calling
"""

from dotenv import load_dotenv
import os
import json
from openai import OpenAI

# Load environment variables
load_dotenv()


class TravelAgentCommands:
    """AutoGPT-style commands for travel planning agents."""

    def __init__(self):
        self.commands = {
            "search_destinations": {
                "function": self.search_destinations,
                "description": "Search for travel information about a destination",
                "parameters": {
                    "destination": {"type": "string", "description": "The destination to research"}
                },
                "required": ["destination"]
            },
            "check_availability": {
                "function": self.check_availability,
                "description": "Check hotel and flight availability",
                "parameters": {
                    "destination": {"type": "string", "description": "The destination"},
                    "dates": {"type": "string", "description": "Travel dates"}
                },
                "required": ["destination", "dates"]
            }
        }

    def search_destinations(self, destination: str) -> str:
        """Search for travel information about a destination."""
        destinations = {
            "Paris": "Paris offers the Eiffel Tower, Louvre Museum, and Notre-Dame. Best time: April-June. Average cost: $200/day.",
            "Tokyo": "Tokyo features temples, cherry blossoms, and modern tech. Best time: March-May. Average cost: $180/day.",
            "Bali": "Bali has beaches, temples, and rice terraces. Best time: April-October. Average cost: $100/day."
        }
        return destinations.get(destination, f"Information about {destination}: Beautiful destination with rich culture.")

    def check_availability(self, destination: str, dates: str) -> str:
        """Check hotel and flight availability."""
        return f"Available for {destination} during {dates}: Hotels from $100/night, Flights from $500 roundtrip."

    def execute_command(self, command_name: str, **kwargs) -> str:
        """Execute a command by name."""
        if command_name in self.commands:
            return self.commands[command_name]["function"](**kwargs)
        return f"Unknown command: {command_name}"

    def get_commands_schema(self) -> list:
        """Get OpenAI function calling schema for commands."""
        schemas = []
        for cmd_name, cmd_info in self.commands.items():
            schema = {
                "name": cmd_name,
                "description": cmd_info["description"],
                "parameters": {
                    "type": "object",
                    "properties": cmd_info["parameters"],
                    "required": cmd_info.get("required", [])
                }
            }
            schemas.append(schema)
        return schemas


class SimpleAutoGPTAgent:
    """Simplified AutoGPT-style agent for multi-agent orchestration."""

    def __init__(self, api_key: str, role: str, commands: TravelAgentCommands):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-3.5-turbo"
        self.role = role
        self.commands = commands
        self.conversation_history = []

    def chat(self, user_message: str) -> str:
        """Process a user message with tool calling."""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        system_message = {
            "role": "system",
            "content": f"You are a {self.role}. Use the available commands to answer questions."
        }

        messages = [system_message] + self.conversation_history

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            functions=self.commands.get_commands_schema(),
            function_call="auto"
        )

        message = response.choices[0].message

        if message.function_call:
            function_name = message.function_call.name
            function_args = json.loads(message.function_call.arguments)

            result = self.commands.execute_command(function_name, **function_args)

            self.conversation_history.append({
                "role": "function",
                "name": function_name,
                "content": result
            })

            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=[system_message] + self.conversation_history
            )

            final_message = final_response.choices[0].message.content
            self.conversation_history.append({
                "role": "assistant",
                "content": final_message
            })

            return final_message
        else:
            content = message.content
            self.conversation_history.append({
                "role": "assistant",
                "content": content
            })
            return content


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        exit(1)

    # Create shared command registry
    commands = TravelAgentCommands()

    # Create specialized agents
    researcher = SimpleAutoGPTAgent(api_key, "Travel Researcher", commands)
    booking_agent = SimpleAutoGPTAgent(api_key, "Booking Specialist", commands)
    itinerary_planner = SimpleAutoGPTAgent(api_key, "Itinerary Planner", TravelAgentCommands())

    # Define travel planning input
    destination = "Paris"
    dates = "June 15-22, 2026"
    preferences = "art, history, local cuisine"

    print(f"🌍 AutoGPT Travel Planner\n")
    print(f"Planning trip to: {destination}")
    print(f"Dates: {dates}")
    print(f"Interests: {preferences}\n")
    print("="*60)

    # NOTE: AutoGPT does not have a built-in workflow orchestration library.
    # AutoGPT agents must be orchestrated manually through sequential function calls.

    # Execute agents sequentially
    print("\n🔍 Step 1: Research Agent")
    research_output = researcher.chat(f"Research {destination} focusing on {preferences}.")

    print("\n✈️  Step 2: Booking Agent")
    booking_output = booking_agent.chat(f"Check availability for {destination} during {dates}.")

    print("\n📅 Step 3: Itinerary Planner")
    itinerary_output = itinerary_planner.chat(
        f"Create a 7-day itinerary for {destination} based on:\n\nResearch: {research_output}\n\nBooking: {booking_output}\n\nFocus on {preferences}."
    )

    print("\n" + "="*60)
    print("📋 Final Travel Plan:")
    print("="*60)
    print(itinerary_output)


if __name__ == "__main__":
    main()
