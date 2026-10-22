from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate   
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",)

template=PromptTemplate(
    template="Write a detailed explanation about:\n{topic}",
    input_variables=["topic"],
)

template2=PromptTemplate(
    template="Write a five-line summary about:\n{topic}",
    input_variables=["topic"],
)

chain=template|model|StrOutputParser()
summary_chain=template2|model|StrOutputParser()

try:
    topic=(
        "The Samsung Galaxy S24 Ultra is a premium flagship "
        "smartphone with excellent overall performance."
    )
    response=chain.invoke({"topic": topic})

    print("\nDETAILED RESPONSE:")
    print(response)

    summary=summary_chain.invoke({"topic": response})

    print("\nFIVE-LINE SUMMARY:")
    print(summary)
except Exception as e:
    print(f"An error occurred: {e}")

