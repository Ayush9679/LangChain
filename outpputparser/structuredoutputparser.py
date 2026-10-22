from langchain_classic.output_parsers import (
    StructuredOutputParser,
    ResponseSchema
)

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate   
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-3.7-flash",)

schema=[
    ResponseSchema(name="summary", description="A concise summary of the topic."),
    ResponseSchema(name="key_points", description="A list of key points related to the topic."),
    ResponseSchema(name="detailed_explanation", description="A detailed explanation of the topic.")
]

parser=StructuredOutputParser.from_response_schemas(schema)


template=PromptTemplate(
    template="Write a detailed explanation about:\n{format_instruction}\n{topic}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()},)
prompt=template.invoke({"topic":"blackhole"})

result=model.invoke(prompt)

final_result=parser.parse(result.text)

print(final_result)
