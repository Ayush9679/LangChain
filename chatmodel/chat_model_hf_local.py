from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
)
model=ChatHuggingFace(llm=llm)

request=model.invoke("What is the husband name of Deepika Padukone?")

print(request.text)