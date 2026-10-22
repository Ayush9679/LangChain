
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

gemini_api_key = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
)

if not gemini_api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing from your .env file."
    )

# Initialize Google Gemini
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=gemini_api_key,
    max_output_tokens=256,
    temperature=0,
)

parser = StrOutputParser()

template = PromptTemplate(
    template="Write a detailed explanation about:\n{topic}",
    input_variables=["topic"],
)

template1 = PromptTemplate(
    template="Write a five-line summary about:\n{topic}",
    input_variables=["topic"],
)

# Build the original chains
chain = template | model | parser
summary_chain = template1 | model | parser

topic = (
    "The Samsung Galaxy S24 Ultra is a premium flagship "
    "smartphone with excellent overall performance."
)

try:
    response = chain.invoke({"topic": topic})

    print("\nDETAILED RESPONSE:")
    print(response)

    summary = summary_chain.invoke({"topic": response})

    print("\nFIVE-LINE SUMMARY:")
    print(summary)

except Exception as e:
    print(f"\nGoogle Gemini inference failed: {e}")
