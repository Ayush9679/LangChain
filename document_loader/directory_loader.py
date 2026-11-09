from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
model=ChatGoogleGenerativeAI(model="gemini-3.7-flash")
loader=DirectoryLoader(
    path="../data",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
docs=loader.load()
print(docs[0].page_content)
print(len(docs))
print(docs[0].metadata)

docs1=loader.lazy_load()

for d in docs1:
    print(d.page_content)
    print(d.metadata)