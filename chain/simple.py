from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import chain  
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.7-flash")

prompt=PromptTemplate(
    template="Write a detailed explanation about:\n{topic}",
    input_variables=["topic"],
)   

chain=prompt|model|StrOutputParser()

result=chain.invoke({"topic":"Taarak mehta ka oolta chashma"})
print(result)