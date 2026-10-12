from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv  
from langchain_core.prompts import PromptTemplate
load_dotenv()
llm=HuggingFaceEndpoint(repo_id="TinyLlama/TinyLlama-2-7B-GGUF", task="text-generation")

model=ChatHuggingFace(llm=llm)

template=PromptTemplate(
    template="Write a detailed based over /n {topic}",
    input_variables=["topic"]
)

template1=PromptTemplate(
    template="Write a 5 line summary based over /n {topic}",
    input_variables=["topic"]
)   

template.invoke({"topic":"The Samsung Galaxy S24 Ultra is a premium flagship smartphone with excellent overall performance."})

response=model.invoke(template)

template1.invoke({"topic":response.text})

print(model.invoke(template1))

        

