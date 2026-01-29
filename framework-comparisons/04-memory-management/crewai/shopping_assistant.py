# Muneeb Ahmad | https://github.com/muneebsa/agentic-frameworks-lab | MIT License | Educational purposes only

"""
CrewAI Shopping Assistant - Memory Management with Multi-Agent Workflow
Demonstrates: 2-agent system with short-term, long-term (persistent), and summarization memory
"""

import os
import json
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

load_dotenv()

# Configure LLM
llm = LLM(model="gpt-3.5-turbo", temperature=0.7)

# Agent 1: Memory Manager (persistent long-term storage)
PROFILE_FILE = os.path.join(os.path.dirname(__file__), "customer_profile.json")

def memory_manager_load_profile() -> dict:
    """Load customer profile from disk (permanent storage)."""
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, 'r') as f:
            return json.load(f)
    return {
        "name": "Sarah",
        "preferences": ["good camera", "long battery life"],
        "past_purchases": ["iPhone 12 (2022)", "AirPods Pro (2023)"],
        "budget_history": ["$800-$1200"]
    }

def memory_manager_save_profile(profile: dict):
    """Save customer profile to disk (permanent storage)."""
    with open(PROFILE_FILE, 'w') as f:
        json.dump(profile, f, indent=2)

customer_profile = memory_manager_load_profile()

@tool
def get_customer_profile() -> str:
    """Memory Manager Agent: Retrieve customer profile (long-term memory)."""
    return f"Customer: {customer_profile['name']}\nPreferences: {', '.join(customer_profile['preferences'])}\nPast purchases: {', '.join(customer_profile['past_purchases'])}\nTypical budget: {customer_profile['budget_history'][-1]}"

def main():
    print("🛒 CrewAI Shopping Assistant (2-Agent Memory)\n" + "=" * 60)
    print("Memory Types: Short-term (buffer), Long-term (persistent file), Summarization")
    print(f"📁 Loaded profile from: {PROFILE_FILE}")
    print(f"👤 Customer: {customer_profile['name']}, Preferences: {customer_profile['preferences']}\n")

    # Agent 2: Shopping Assistant (uses Memory Manager tool)
    shopping_agent = Agent(
        role="Shopping Assistant",
        goal="Help customers find products using their profile and conversation history",
        backstory="Friendly TechStore assistant with access to customer profiles.",
        tools=[get_customer_profile],
        llm=llm,
        verbose=False
    )

    # Short-term memory (conversation buffer)
    conversation_buffer = []

    # Simulated conversation
    queries = [
        "I need a new phone",
        "What's my budget usually?",
        "I want something under $1000 this time",
        "Does it have a good camera like my preferences show?"
    ]

    for i, query in enumerate(queries, 1):
        print(f"{'='*60}\n💬 Customer (Turn {i}): {query}\n{'='*60}")

        # Build short-term context
        context = "\n".join([f"Turn {j}: Customer: {q}" for j, q in enumerate(conversation_buffer[-3:], 1)])

        task = Task(
            description=f"Previous conversation:\n{context}\n\nCurrent: {query}\n\nUse get_customer_profile tool for personalized help.",
            agent=shopping_agent,
            expected_output="Helpful personalized response"
        )

        crew = Crew(agents=[shopping_agent], tasks=[task], verbose=False)
        result = crew.kickoff()
        print(f"🤖 Shopping Assistant: {result}\n")

        conversation_buffer.append(query)

        # Detect and store new preferences
        if "under $1000" in query.lower():
            customer_profile['preferences'].append("budget-conscious")
            memory_manager_save_profile(customer_profile)
            print(f"💾 [Memory Manager] Saved new preference to {PROFILE_FILE}: budget-conscious\n")

    # Summarization
    print(f"{'='*60}\n📝 Conversation Summary:\n{'='*60}")
    summary_task = Task(
        description=f"Summarize in 2-3 sentences: {' / '.join(queries)}",
        agent=shopping_agent,
        expected_output="Brief summary"
    )
    summary_crew = Crew(agents=[shopping_agent], tasks=[summary_task], verbose=False)
    print(f"{summary_crew.kickoff()}\n")

    # Show memories
    print(f"🧠 Short-term: {len(conversation_buffer)} messages")
    print(f"💾 Permanent: {PROFILE_FILE} - {customer_profile['preferences']}\n")

if __name__ == "__main__":
    main()
