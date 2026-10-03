from anthropic.types import model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

documents=[
    "Virat Kohli is an Indian international cricketer and one of the most recognized batsmen of his generation. He has represented the Indian cricket team across all major formats and has served as its captain. Kohli is particularly known for his aggressive batting style, consistency, fitness, and ability to perform under pressure. He has scored numerous international centuries and has achieved several records in both international and domestic cricket. Beyond cricket, Kohli is also known for his fitness-focused lifestyle and charitable activities."
]
query="Tell me about virat kohli"

embed_model=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

doc_embed=embed_model.embed_documents(documents)
query_embed=embed_model.embed_query(query)

similarity=cosine_similarity([query_embed],doc_embed)
print(similarity)