
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableSequence,
    RunnablePassthrough,
    RunnableBranch,
    RunnableLambda
)

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

# Generate detailed report
prompt1 = PromptTemplate(
    template="Generate a detailed report based on {topic}",
    input_variables=["topic"]
)

# Generate summary
prompt2 = PromptTemplate(
    template="Summarize the following report:\n\n{report}",
    input_variables=["report"]
)

parser1 = StrOutputParser()

gen_chain = RunnableSequence(prompt1, model, parser1)

# Convert string into dictionary for prompt2
summary_chain = (
    RunnableLambda(lambda x: {"report": x})
    | prompt2
    | model
    | parser1
)

# Conditional execution
decision_chain = RunnableBranch(
    (
        lambda x: len(x.split()) > 100,
        summary_chain
    ),
    RunnablePassthrough()  # Default branch
)

# Combine everything
final_chain = gen_chain | decision_chain

result = final_chain.invoke({
    "topic": "AI in healthcare"
})

print(result)
