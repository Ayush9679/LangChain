from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

chat_history = []

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    # Store user's message
    chat_history.append(
        HumanMessage(content=user_input)
    )

    # Send complete conversation to Gemini
    response = model.invoke(chat_history)

    # Store AI's response
    chat_history.append(
        AIMessage(content=response.content)
    )

    print("AI:", response.text)