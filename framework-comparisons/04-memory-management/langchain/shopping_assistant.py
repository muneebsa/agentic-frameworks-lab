# Muneeb Ahmad | https://github.com/muneebsa/agentic-frameworks-lab | MIT License | Educational purposes only

"""
LangChain Shopping Assistant - Memory Management with Multi-Agent Workflow
Demonstrates: 2-agent system with short-term, long-term (persistent), and summarization memory
"""

import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Agent 1: Memory Manager (persistent long-term storage)
PROFILE_FILE = os.path.join(os.path.dirname(__file__), "customer_profile.json")

def memory_manager_load_profile() -> dict:
    """Load customer profile from disk (permanent storage)."""
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, 'r') as f:
            return json.load(f)
    # Default profile for first-time customer
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

# Load persistent profile
customer_profile = memory_manager_load_profile()

def memory_manager_get_profile() -> str:
    """Memory Manager Agent: Retrieve customer profile (long-term memory)."""
    return f"Customer: {customer_profile['name']}\nPreferences: {', '.join(customer_profile['preferences'])}\nPast purchases: {', '.join(customer_profile['past_purchases'])}\nTypical budget: {customer_profile['budget_history'][-1]}"

def memory_manager_store_preference(preference: str):
    """Memory Manager Agent: Store new preference and persist to disk."""
    if preference not in customer_profile['preferences']:
        customer_profile['preferences'].append(preference)
        memory_manager_save_profile(customer_profile)  # Save to disk

# Agent 2: Shopping Assistant (short-term conversation buffer)
short_term_memory = []  # List of messages (HumanMessage, AIMessage)

def main():
    print("🛒 LangChain Shopping Assistant (2-Agent Memory)\n" + "=" * 60)
    print("Memory Types: Short-term (buffer), Long-term (persistent file), Summarization")
    print(f"📁 Loaded profile from: {PROFILE_FILE}")
    print(f"👤 Customer: {customer_profile['name']}, Preferences: {customer_profile['preferences']}\n")

    # Shopping Assistant prompt
    assistant_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a TechStore shopping assistant. Customer profile:\n{profile}\n\nUse this info to provide personalized recommendations."),
        ("human", "{query}")
    ])
    assistant_chain = assistant_prompt | llm | StrOutputParser()

    # Simulated conversation
    queries = [
        "I need a new phone",
        "What's my budget usually?",
        "I want something under $1000 this time",
        "Does it have a good camera like my preferences show?"
    ]

    for i, query in enumerate(queries, 1):
        print(f"{'='*60}\n💬 Customer (Turn {i}): {query}\n{'='*60}")

        # Step 1: Memory Manager provides long-term profile
        profile = memory_manager_get_profile()

        # Step 2: Shopping Assistant uses profile + short-term memory
        response = assistant_chain.invoke({"profile": profile, "query": query})
        print(f"🤖 Shopping Assistant: {response}\n")

        # Step 3: Store in short-term memory
        short_term_memory.append(HumanMessage(content=query))
        short_term_memory.append(AIMessage(content=response))

        # Step 4: Detect new preferences and persist to disk
        if "under $1000" in query.lower():
            memory_manager_store_preference("budget-conscious")
            print(f"💾 [Memory Manager] Saved new preference to {PROFILE_FILE}: budget-conscious\n")

    # Demonstrate summarization memory (compress conversation)
    print(f"{'='*60}\n📝 Conversation Summary (Summarization Memory):\n{'='*60}")
    conversation_text = "\n".join([f"{'Customer' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" for msg in short_term_memory])
    summary_prompt = f"Summarize this conversation in 2-3 sentences:\n\n{conversation_text}"
    summary = llm.invoke(summary_prompt).content
    print(f"{summary}\n")

    # Show short-term memory
    print(f"{'='*60}\n🧠 Short-term Memory (Last {len(short_term_memory)} messages):\n{'='*60}")
    for msg in short_term_memory[-4:]:
        msg_type = "CUSTOMER" if isinstance(msg, HumanMessage) else "ASSISTANT"
        print(f"{msg_type}: {msg.content[:100]}...")

    # Show permanent memory
    print(f"\n{'='*60}\n💾 Permanent Memory (Persists across sessions):\n{'='*60}")
    print(f"File: {PROFILE_FILE}")
    print(f"Current preferences: {customer_profile['preferences']}")
    print("ℹ️  This profile will be loaded when you run the program again.\n")

if __name__ == "__main__":
    main()
