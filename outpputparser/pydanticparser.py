
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import PydanticOutputParser

load_dotenv()


# 1. Define the expected output structure
class Person(BaseModel):
    name: str = Field(description="The name of the fictional person.")
    city: str = Field(description="The city where the person lives.")
    age: int = Field(
        gt=12,
        lt=100,
        description="Age must be between 13 and 99."
    )


# 2. Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-3.7-flash")


# 3. Initialize the parser
parser = PydanticOutputParser(pydantic_object=Person)


# 4. Include formatting instructions in the prompt
template = PromptTemplate(
    template=(
        "Generate one fictional person's profile.\n"
        "Follow the required output format exactly.\n"
        "{format_instructions}\n"
        "Topic: {topic}"
    ),
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)


# 5. Create the chain
person_chain = template | model | parser


# 6. Execute the chain
result = person_chain.invoke({
    "topic": "Generate a fictional individual living in Delhi."
})


# 7. Print the validated Pydantic object
print(result)

# Access individual fields
print("Name:", result.name)
print("City:", result.city)
print("Age:", result.age)

# Convert the object into a Python dictionary
print(result.model_dump())
