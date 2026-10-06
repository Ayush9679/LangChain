from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

messages=[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hello! How are you?")
]

response = model.invoke(messages)

messages.append(AIMessage(content=response.text))

print("AI:", response.text)
print(messages)