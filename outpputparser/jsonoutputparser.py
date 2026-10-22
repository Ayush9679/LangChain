from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import chain

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",)

parser=JsonOutputParser()

template=PromptTemplate(
    template="Write a detailed explanation about:\n{format_instruction}\n{topic}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

chain=template|model|parser

res=chain.invoke({"topic": "The Samsung Galaxy S24 Ultra is a premium flagship smartphone with excellent overall performance."})
print(res)
