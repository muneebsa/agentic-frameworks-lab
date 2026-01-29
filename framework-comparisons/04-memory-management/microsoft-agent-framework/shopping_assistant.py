# Muneeb Ahmad | https://github.com/muneebsa/agentic-frameworks-lab | MIT License | Educational purposes only

"""
Microsoft Agent Framework Shopping Assistant - Memory Management with Multi-Agent Workflow
Demonstrates: 2-agent system with short-term, long-term (persistent), and summarization memory
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

def memory_manager_get_profile() -> str:
    """Memory Manager Agent: Retrieve customer profile (long-term memory)."""
    return f"Customer: {customer_profile['name']}\nPreferences: {', '.join(customer_profile['preferences'])}\nPast purchases: {', '.join(customer_profile['past_purchases'])}\nTypical budget: {customer_profile['budget_history'][-1]}"

def main():
    print("🛒 Microsoft Agent Framework Shopping Assistant (2-Agent Memory)\n" + "=" * 60)
    print("Memory Types: Short-term (buffer), Long-term (persistent file), Summarization")
    print(f"📁 Loaded profile from: {PROFILE_FILE}")
    print(f"👤 Customer: {customer_profile['name']}, Preferences: {customer_profile['preferences']}\n")

    # Agent 2: Shopping Assistant with short-term memory
    short_term_memory = []

    # Simulated conversation
    queries = [
        "I need a new phone",
        "What's my budget usually?",
        "I want something under $1000 this time",
        "Does it have a good camera like my preferences show?"
    ]

    for i, query in enumerate(queries, 1):
        print(f"{'='*60}\n💬 Customer (Turn {i}): {query}\n{'='*60}")

        # Step 1: Memory Manager provides profile
        profile = memory_manager_get_profile()

        # Step 2: Shopping Assistant responds
        messages = [
            {"role": "system", "content": f"You are a TechStore shopping assistant. Customer profile:\n{profile}"},
            {"role": "user", "content": query}
        ]
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, temperature=0.7)
        assistant_reply = response.choices[0].message.content
        print(f"🤖 Shopping Assistant: {assistant_reply}\n")

        # Store in short-term memory
        short_term_memory.append({"user": query, "assistant": assistant_reply})

        # Detect and store new preferences
        if "under $1000" in query.lower():
            customer_profile['preferences'].append("budget-conscious")
            memory_manager_save_profile(customer_profile)
            print(f"💾 [Memory Manager] Saved new preference to {PROFILE_FILE}: budget-conscious\n")

    # Summarization
    print(f"{'='*60}\n📝 Conversation Summary:\n{'='*60}")
    summary_messages = [{"role": "user", "content": f"Summarize in 2-3 sentences:\n{' / '.join(queries)}"}]
    summary_response = client.chat.completions.create(model="gpt-3.5-turbo", messages=summary_messages)
    print(f"{summary_response.choices[0].message.content}\n")

    # Show memories
    print(f"🧠 Short-term: {len(short_term_memory)} messages")
    print(f"💾 Permanent: {PROFILE_FILE} - {customer_profile['preferences']}\n")

if __name__ == "__main__":
    main()
