
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# Step 1: Generate a joke
prompt = PromptTemplate(
    template="Create a joke based on this topic: {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

f_chain = prompt | model | parser

# Step 2: Create a chain to explain the generated joke
explanation_prompt = PromptTemplate(
    template="Explain the following joke in simple terms: {joke}",
    input_variables=["joke"]
)

explanation_chain = explanation_prompt | model | parser

# Step 3: Pass the same joke to two parallel branches
pass_through = RunnablePassthrough()

result_chain = RunnableParallel({
    "joke": pass_through,
    "explanation": explanation_chain
})

# Step 4: Execute sequentially, then in parallel
answer_chain = f_chain | result_chain

final_v = answer_chain.invoke({"topic": "AI in healthcare"})

print(final_v)
