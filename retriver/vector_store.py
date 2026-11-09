from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2"
)

# Step 1: Your source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

vector_store=Chroma.from_documents(documents, embeddings)

retriever=vector_store.as_retriever(search_kwargs={"k": 2}) 

results=retriever.invoke("What is LangChain?")

for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(f"Content: {doc.page_content}\n")

