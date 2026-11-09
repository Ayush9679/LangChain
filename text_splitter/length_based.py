from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
load_dotenv()

loader = TextLoader(file_path="../data/poem.txt")

result=loader.lazy_load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=25,
    chunk_overlap=2
)

split_result = splitter.split_documents(result)

print(len(split_result))
