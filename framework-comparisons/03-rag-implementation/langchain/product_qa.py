# Muneeb Ahmad | https://github.com/muneebsa/agentic-frameworks-lab | MIT License | Educational purposes only

"""
LangChain Customer Support - RAG with Multi-Agent Workflow
Demonstrates: 2-agent system (KB Agent retrieves from vector DB, Support Agent responds to customer)
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Initialize components
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load FAISS index
index_path = os.path.join(os.path.dirname(__file__), "..", "faiss_index")
vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Agent 1: KB Agent (RAG retrieval)
def kb_agent_search(query: str) -> str:
    """Knowledge Base Agent: Retrieve product info from vector DB."""
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = ", ".join({os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in docs})
    return f"Sources: {sources}\n\n{context}"

def main():
    print("📚 LangChain Customer Support (2-Agent RAG)\n" + "=" * 60)

    # Agent 2: Support Agent (customer-facing)
    support_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a TechStore support agent. Use the KB info below to answer:\n\n{kb_result}"),
        ("human", "{query}")
    ])
    support_agent = support_prompt | llm | StrOutputParser()

    # Customer queries
    queries = [
        "What is the camera resolution of the iPhone 15 Pro?",
        "I need a phone with long battery life. Can you compare the options?",
        "Which phone has the best AI features?"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}\n🧑 Customer Query {i}: {query}\n{'='*60}")

        # Step 1: KB Agent retrieves
        print("🔍 [KB Agent] Searching knowledge base...")
        kb_result = kb_agent_search(query)
        print("✓ Retrieved documents\n")

        # Step 2: Support Agent responds
        print("💬 [Support Agent] Response:")
        response = support_agent.invoke({"kb_result": kb_result, "query": query})
        print(f"{response}\n")

if __name__ == "__main__":
    main()
