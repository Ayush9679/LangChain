from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
model=ChatGoogleGenerativeAI(model="gemini-3.7-flash")
Loader=PyPDFLoader("../data/file-sample.pdf")
docs=Loader.load()
print(docs[0].page_content)