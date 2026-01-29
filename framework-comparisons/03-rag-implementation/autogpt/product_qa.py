# Muneeb Ahmad | https://github.com/muneebsa/agentic-frameworks-lab | MIT License | Educational purposes only

"""
AutoGPT Customer Support - RAG with Multi-Agent Workflow
Demonstrates: 2-agent system (KB Agent retrieves from vector DB, Support Agent responds to customer)
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load pre-built FAISS index
index_path = os.path.join(os.path.dirname(__file__), "..", "faiss_index")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Agent 1: KB Agent function (RAG tool)
def search_product_kb(query: str) -> str:
    """Knowledge Base Agent: Retrieve product info from vector DB."""
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = ", ".join({os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in docs})
    return f"Sources: {sources}\n\n{context}"

# Define function for OpenAI
kb_function = [{
    "type": "function",
    "function": {
        "name": "search_product_kb",
        "description": "Search product knowledge base for smartphone info",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    }
}]

def run_support_agent(customer_query: str) -> str:
    """Agent 2: Support Agent with KB access."""
    messages = [
        {"role": "system", "content": "You are a TechStore support agent. Use search_product_kb to get info."},
        {"role": "user", "content": customer_query}
    ]

    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, tools=kb_function, tool_choice="auto")
    response_message = response.choices[0].message

    if response_message.tool_calls:
        messages.append(response_message)
        for tool_call in response_message.tool_calls:
            if tool_call.function.name == "search_product_kb":
                result = search_product_kb(query=json.loads(tool_call.function.arguments).get("query"))
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": "search_product_kb", "content": result})

        final_response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
        return final_response.choices[0].message.content

    return response_message.content

def main():
    print("📚 AutoGPT Customer Support (2-Agent RAG)\n" + "=" * 60)

    queries = [
        "What is the camera resolution of the iPhone 15 Pro?",
        "I need a phone with long battery life. Can you compare the options?",
        "Which phone has the best AI features?"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}\n🧑 Customer Query {i}: {query}\n{'='*60}")
        response = run_support_agent(query)
        print(f"💬 Support Agent: {response}\n")

if __name__ == "__main__":
    main()
