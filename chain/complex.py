
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import (
    RunnableParallel,
    RunnableBranch, RunnableLambda)
from dotenv import load_dotenv

load_dotenv()

# 1. Initialize Gemini with a supported model
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)

# 2. Define the expected output structure
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Classify the text as positive or negative"
    )

# 3. Create the parser
parser = PydanticOutputParser(
    pydantic_object=Feedback
)

# 4. Add formatting instructions to the prompt
temp = PromptTemplate(
    template="""
Classify this sentence as positive or negative: {topic}

{format_instructions}
""",
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

prompt2=PromptTemplate(
    template="Write an appropriate response based over the positive feedback: {topic}",
    input_variables=["topic"]
)

prompt3=PromptTemplate(
    template="Write an appropriate response based over the negative feedback:{topic}",
    input_variables=["topic"]
)

# 5. Create the LCEL chain
classifier_chain = temp | model | parser

# 6. Invoke the chain
result = classifier_chain.invoke({
    "topic": "I hate this smartphone"
})

branch_chain=RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model),
    (lambda x: x.sentiment == "negative", prompt3 | model),
    RunnableLambda(lambda x: "No feedback provided"))

final_chain=classifier_chain | branch_chain

results=final_chain.invoke({"topic": "I hate this smartphone"})

print("Final Chain Result:",results.text)

print(result)
print(result.sentiment)
