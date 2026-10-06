from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessage(content="{input}")
])

chat_history = []

with open("chat_history.txt", "a+") as file:
    file.seek(0)

    for line in file:
        line = line.strip()

        if line.startswith("Human:"):
            chat_history.append(
                HumanMessage(content=line.replace("Human:", "").strip())
            )

        elif line.startswith("AI:"):
            chat_history.append(
                AIMessage(content=line.replace("AI:", "").strip())
            )

print(chat_history)

user_input = "I wish you to know my name is Ayush Dubey"

prompt = chat_template.invoke({
    "chat_history": chat_history,
    "input": user_input
})

response = model.invoke(prompt.messages)

chat_history.append(HumanMessage(content=user_input))
chat_history.append(AIMessage(content=response.text))

with open("chat_history.txt", "a") as file:
    file.write(f"Human: {user_input}\n")
    file.write(f"AI: {response.text}\n")

print("AI:", response.text)