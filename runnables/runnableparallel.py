from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel
load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.3)

prompt=PromptTemplate(
    template="create a linkedin post based over this {topic}",
    input_variables=["topic"]
)
parser=StrOutputParser()
prompt1=PromptTemplate(
    template="create a instagram post based over this {topic}",
    input_variables=["topic"]
)

parser1=StrOutputParser()

parallel_runnable = RunnableParallel({"linkedin": RunnableSequence(prompt, model, parser), "instagram": RunnableSequence(prompt1, model, parser1)})

parallel_result = parallel_runnable.invoke({"topic": "AI in healthcare"})


print(parallel_result)
