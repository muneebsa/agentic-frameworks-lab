# Muneeb Ahmad | https://github.com/muneebsa/agentic-frameworks-lab | MIT License | Educational purposes only

"""
LlamaIndex Customer Support - RAG with Multi-Agent Workflow
Demonstrates: 2-agent system (KB Agent retrieves from vector DB, Support Agent responds to customer)
"""

import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Configure settings
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Load shared FAISS index (built by LangChain, used across all frameworks)
index_path = os.path.join(os.path.dirname(__file__), "..", "faiss_index")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Wrap LangChain retriever for LlamaIndex
def kb_agent_search(query: str) -> str:
    """Knowledge Base Agent: Search product knowledge base using shared FAISS index."""
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = ", ".join({os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in docs})
    return f"Sources: {sources}\n\n{context}"

def main():
    print("📚 LlamaIndex Customer Support (2-Agent RAG)\n" + "=" * 60)

    queries = [
        "What is the camera resolution of the iPhone 15 Pro?",
        "I need a phone with long battery life. Can you compare the options?",
        "Which phone has the best AI features?"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}\n🧑 Customer Query {i}: {query}\n{'='*60}")

        # Step 1: KB Agent retrieves from shared FAISS index
        print("🔍 [KB Agent] Searching knowledge base...")
        kb_result = kb_agent_search(query)
        print("✓ Retrieved documents\n")

        # Step 2: Support Agent responds using LlamaIndex LLM
        print("💬 [Support Agent] Response:")
        prompt = f"You are a TechStore support agent. Use this info to answer:\n\n{kb_result}\n\nCustomer: {query}\n\nAnswer:"
        response = Settings.llm.complete(prompt)
        print(f"{response.text}\n")

if __name__ == "__main__":
    main()
