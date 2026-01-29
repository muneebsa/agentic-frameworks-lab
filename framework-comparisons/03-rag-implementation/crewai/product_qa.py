# Muneeb Ahmad | https://github.com/muneebsa/agentic-frameworks-lab | MIT License | Educational purposes only

"""
CrewAI Customer Support - RAG with Multi-Agent Workflow
Demonstrates: 2-agent system (KB Agent retrieves from vector DB, Support Agent responds to customer)
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Load pre-built FAISS index (shared across all frameworks)
index_path = os.path.join(os.path.dirname(__file__), "..", "faiss_index")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Agent 1: KB Agent tool (vector search)
@tool
def search_knowledge_base(query: str) -> str:
    """Knowledge Base Agent: Search product knowledge base using vector search."""
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = ", ".join({os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in docs})
    return f"Sources: {sources}\n\n{context}"

def main():
    llm = LLM(model="gpt-3.5-turbo", temperature=0.7)
    print("📚 CrewAI Customer Support (2-Agent RAG)\n" + "=" * 60)

    # Agent 2: Support Agent (uses KB Agent tool)
    support_agent = Agent(
        role="Customer Support Agent",
        goal="Help customers with smartphone questions using the knowledge base",
        backstory="Friendly TechStore support agent who consults product docs.",
        tools=[search_knowledge_base],
        llm=llm,
        verbose=False
    )

    queries = [
        "What is the camera resolution of the iPhone 15 Pro?",
        "I need a phone with long battery life. Can you compare the options?",
        "Which phone has the best AI features?"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}\n🧑 Customer Query {i}: {query}\n{'='*60}")

        task = Task(
            description=f"Answer: {query}",
            agent=support_agent,
            expected_output="Helpful answer with product details"
        )

        crew = Crew(agents=[support_agent], tasks=[task], verbose=False)
        result = crew.kickoff()
        print(f"💬 Support Agent: {result}\n")

if __name__ == "__main__":
    main()
