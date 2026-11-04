
from dotenv import load_dotenv

from langchain_classic.chains import RetrievalQA
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

load_dotenv()

# 1. Load the poem
loader = TextLoader("../data/poem.txt", encoding="utf-8")
documents = loader.load()

print(f"Documents loaded: {len(documents)}")

# 2. Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)

texts = text_splitter.split_documents(documents)

print(f"Chunks created: {len(texts)}")

# 3. Initialize the embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# 4. Store embeddings in Chroma
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    collection_name="poem_collection",
)

print("Vector database created successfully!")

# 5. Create the retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# 6. Initialize the Gemini chat model
model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
)

# 7. Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever,
    chain_type="stuff",
)

# 8. Ask a question
query = (
    "What does the lantern symbolize in the poem, "
    "and how does it help the traveler during the rain?"
)

result = qa_chain.invoke({"query": query})

# 9. Display the answer
print("\nQuestion:", query)
print("\nAnswer:", result["result"])