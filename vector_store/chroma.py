from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_core.documents import Document
load_dotenv()

doc1=Document(page_content="Justin Bieber is one of the most prominent singer", metadata={"source": "doc1"})
doc2=Document(page_content="MSD is one of the worlds best wickekeeper", metadata={"source": "doc2"})
doc3=Document(page_content="Tom Holland potrayed as the spiderman", metadata={"source": "doc3"})

docs=[doc1, doc2, doc3]

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vector_store=Chroma(
    embedding_function=embeddings,
    persist_directory="chroma_db",
    collection_name="my_collection"
)

vector_store.add_documents(docs)

x=vector_store.get(include=[ "embeddings", "metadatas","documents"])
print(x)

z=vector_store.similarity_search("who is the best wicketkeeper", k=1)
print(z)