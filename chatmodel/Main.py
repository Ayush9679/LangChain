from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

chat = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    max_tokens=128,
)

response = chat.invoke("lyrics of the song 'Shape of You' by Ed Sheeran")

print(response.content)