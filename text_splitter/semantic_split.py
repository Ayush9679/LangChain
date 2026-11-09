from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

# Load document
loader = TextLoader(
    file_path="../data/poem.txt",
    encoding="utf-8"
)

docs = loader.load()

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# Initialize semantic chunker
chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.0,
)

# Split documents
split_result = chunker.split_documents(docs)

# Display results
print("Total chunks:", len(split_result))

for i, chunk in enumerate(split_result):
    print(f"\nChunk {i+1}:")
    print(chunk.page_content)
    print("Metadata:", chunk.metadata)