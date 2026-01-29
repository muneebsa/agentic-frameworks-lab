# Muneeb Ahmad | https://github.com/muneebsa/agentic-frameworks-lab | MIT License | Educational purposes only

"""
AutoGPT Weather Assistant - Tool Calling Example
Demonstrates: Autonomous agents, command-based tools, experimental features
"""

from dotenv import load_dotenv
import os
import json

# AutoGPT setup - note this is a simplified adaptation
# Full AutoGPT requires more setup; this shows the conceptual approach
try:
    from openai import OpenAI
except ImportError:
    print("OpenAI package not installed. Install with: pip install openai")
    exit(1)

# Load environment variables
load_dotenv()


class WeatherCommand:
    """AutoGPT-style command for weather operations."""

    def __init__(self):
        self.commands = {
            "get_weather": {
                "function": self.get_weather,
                "description": "Get current weather for a city",
                "parameters": {
                    "city": {"type": "string", "description": "The city to get weather for"}
                },
                "required": ["city"]
            }
        }

    def get_weather(self, city: str) -> str:
        """Get current weather for a city."""
        weather_data = {
            "San Francisco": "72°F, Sunny",
            "New York": "65°F, Cloudy",
            "Seattle": "58°F, Rainy",
            "Miami": "85°F, Sunny"
        }
        return weather_data.get(city, f"Weather data not available for {city}")

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
    """Simplified AutoGPT-style agent for tool calling."""

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-3.5-turbo"
        self.commands = WeatherCommand()
        self.conversation_history = []

    def chat(self, user_message: str) -> str:
        """Process a user message with tool calling."""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # System message
        system_message = {
            "role": "system",
            "content": "You are a helpful weather assistant. Use the available commands to answer questions."
        }

        # Get completion with function calling
        messages = [system_message] + self.conversation_history

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            functions=self.commands.get_commands_schema(),
            function_call="auto"
        )

        message = response.choices[0].message

        # Check if function was called
        if message.function_call:
            # Execute the function
            function_name = message.function_call.name
            function_args = json.loads(message.function_call.arguments)

            print(f"→ Calling {function_name} with {function_args}")

            result = self.commands.execute_command(function_name, **function_args)

            # Add function result to conversation
            self.conversation_history.append({
                "role": "function",
                "name": function_name,
                "content": result
            })

            # Get final response
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
            # Direct response without function call
            content = message.content
            self.conversation_history.append({
                "role": "assistant",
                "content": content
            })
            return content


def main():
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        exit(1)

    # Create agent
    agent = SimpleAutoGPTAgent(api_key)

    # Test queries
    test_queries = [
        "What's the weather in San Francisco?",
        "What's the weather in New York?",
        "Tell me about the weather in Seattle and Miami"
    ]

    print("🌤️  AutoGPT Weather Assistant\n")
    print("Note: This is a simplified AutoGPT-style implementation\n")

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        response = agent.chat(query)
        print(f"\nResponse: {response}\n")


if __name__ == "__main__":
    main()
