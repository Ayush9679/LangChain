from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv 
load_dotenv()

llm=ChatGoogleGenerativeAI(model="gemini-3.5-flash",temperature=0.2)
response=llm.invoke("Write a poem about the moon in the style of Shakespeare.")
print(response.text)