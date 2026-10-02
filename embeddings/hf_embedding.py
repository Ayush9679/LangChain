from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

embed_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector=embed_model.embed_query("What is the husband name of Deepika Padukone?")
print(vector)
print(len(vector))