from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-3.7-flash")

loader=TextLoader("../data/poem.txt")

documents=loader.load()

print(documents[0].page_content)



