from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain

from dotenv import load_dotenv


load_dotenv()

llm=ChatGoogleGenerativeAI(model="gemini-3.7-flash",temperature=1)

prompt=PromptTemplate(
    template="Write a poem about {topic}",
    input_variables=["topic"]
)

topic=input("Enter a topic for the poem: ")

formatted_prompt=prompt.format(topic=topic)

chain=LLMChain(llm=llm,prompt=prompt)

poem_topic=chain.run(formatted_prompt)


print(poem_topic.text)


