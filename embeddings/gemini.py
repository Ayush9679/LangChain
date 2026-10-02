from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",dimensions=32)

vector=embedding.embed_query("What is the husband name of Deepika Padukone?",output_dimensionality=32)
print(vector)
print(len(vector))