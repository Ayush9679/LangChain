
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct-1M",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
    max_new_tokens=256,
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

template = PromptTemplate(
    template="Write a detailed explanation about:\n{topic}",
    input_variables=["topic"],
)

template1 = PromptTemplate(
    template="Write a 5-line summary about:\n{topic}",
    input_variables=["topic"],
)

chain = template | model | parser
summary_chain = template1 | model | parser

topic = (
    "The Samsung Galaxy S24 Ultra is a premium flagship "
    "smartphone with excellent overall performance."
)

response = chain.invoke({"topic": topic})

print("DETAILED RESPONSE:")
print(response)

summary = summary_chain.invoke({"topic": response})

print("\nFIVE-LINE SUMMARY:")
print(summary)
