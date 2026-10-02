from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

token = os.getenv("HF_TOKEN")

print("Token loaded:", bool(token))

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    provider="groq",
    task="text-generation",
    huggingfacehub_api_token=token,
    max_new_tokens=100,
)

chat = ChatHuggingFace(llm=llm)

response = chat.invoke("Why is the sky blue?")

print(response.content)