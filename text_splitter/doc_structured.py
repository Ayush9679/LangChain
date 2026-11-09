from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
load_dotenv()

loader = TextLoader(file_path="../data/py.txt")

result=loader.lazy_load()

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=50,
    chunk_overlap=2
)

split_result = splitter.split_documents(result)

print(split_result[0].page_content)
