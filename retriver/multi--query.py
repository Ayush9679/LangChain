from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_classic.retrievers import MultiQueryRetriever
from dotenv import load_dotenv
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2")

all_docs=[
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="FAISS is a library for efficient similarity search and clustering of dense vectors."),
    Document(page_content="MMR (Maximal Marginal Relevance) is a technique for diverse retrieval."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

vector_store=FAISS.from_documents(all_docs, embeddings)

retriever=vector_store.as_retriever(search_kwargs={"k":2},search_type="similarity")
multi_retriever=MultiQueryRetriever.from_llm(llm=ChatGoogleGenerativeAI(model="gemini-3.7-flash"),retriever=retriever)
query="What is LangChain?"
results=multi_retriever.invoke(query)
sr=retriever.invoke(query)

for i in range(len(results)):
    print(f"\nResult {i+1}:")
    print(f"Content: {results[i].page_content}\n")

for i in range(len(sr)):
    print(f"\nSimilarity Search Result {i+1}:")
    print(f"Content: {sr[i].page_content}\n")