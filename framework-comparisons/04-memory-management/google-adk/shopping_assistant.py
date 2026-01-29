# Muneeb Ahmad | https://github.com/muneebsa/agentic-frameworks-lab | MIT License | Educational purposes only

"""
Google ADK Shopping Assistant - Memory Management with Multi-Agent Workflow
Demonstrates: 2-agent system with short-term, long-term (persistent), and summarization memory
"""

import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash-exp")

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
    print("🛒 Google ADK Shopping Assistant (2-Agent Memory)\n" + "=" * 60)
    print("Memory Types: Short-term (buffer), Long-term (persistent file), Summarization")
    print(f"📁 Loaded profile from: {PROFILE_FILE}")
    print(f"👤 Customer: {customer_profile['name']}, Preferences: {customer_profile['preferences']}\n")

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

        # Step 1: Memory Manager provides profile
        profile = memory_manager_get_profile()

        # Step 2: Shopping Assistant responds
        prompt = f"You are a TechStore shopping assistant.\n\nCustomer profile:\n{profile}\n\nRecent conversation: {' / '.join(conversation_buffer[-2:])}\n\nCustomer: {query}\n\nResponse:"
        response = model.generate_content(prompt)
        print(f"🤖 Shopping Assistant: {response.text}\n")

        conversation_buffer.append(query)

        # Detect and store new preferences
        if "under $1000" in query.lower():
            customer_profile['preferences'].append("budget-conscious")
            memory_manager_save_profile(customer_profile)
            print(f"💾 [Memory Manager] Saved new preference to {PROFILE_FILE}: budget-conscious\n")

    # Summarization
    print(f"{'='*60}\n📝 Conversation Summary:\n{'='*60}")
    summary_prompt = f"Summarize this conversation in 2-3 sentences:\n{' / '.join(queries)}"
    summary = model.generate_content(summary_prompt)
    print(f"{summary.text}\n")

    # Show memories
    print(f"🧠 Short-term: {len(conversation_buffer)} messages")
    print(f"💾 Permanent: {PROFILE_FILE} - {customer_profile['preferences']}\n")

if __name__ == "__main__":
    main()
