from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2"
)

docs=[
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="FAISS is a library for efficient similarity search and clustering of dense vectors."),
    Document(page_content="MMR (Maximal Marginal Relevance) is a technique for diverse retrieval.")
]

vector_store=FAISS.from_documents(docs, embeddings)

retriever=vector_store.as_retriever(search_kwargs={"k":2,"lambda_mult":0.5},search_type="mmr")

results=retriever.invoke("What is LangChain?")

for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(f"Content: {doc.page_content}\n") 