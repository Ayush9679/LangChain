import wikipedia

# Identify your application to Wikimedia
wikipedia.set_user_agent(
    "AyushLangChainDemo/1.0 (contact: your-email@example.com)"
)

from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    top_k_results=3,
    lang="en"
)

query = "Who is the best wicketkeeper in cricket?"

results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(f"Title: {doc.metadata['title']}")
    print(f"Content: {doc.page_content}\n")