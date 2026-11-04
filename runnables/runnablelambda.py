from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.3)

prompt=PromptTemplate(
    template="create a linkedin post based over this {topic}",
    input_variables=["topic"])

parser=StrOutputParser()

f_chain=prompt | model | parser

exp_prompt=PromptTemplate(
    template="Explain this {topic}",input_variables=["topic"])

exp_chain=exp_prompt | model | parser


#Runnable lambda

def word_counter(text: str) -> int:
    return len(text.split())
runnable_word_counter = RunnableLambda(word_counter)

second_chain=RunnableParallel({
"explanation": exp_chain,
"word_count": runnable_word_counter
})

result_chain=f_chain | second_chain

answer=result_chain.invoke({"topic": "AI in healthcare"})

print(answer)

