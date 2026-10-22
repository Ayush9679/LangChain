
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

# 1. Initialize models
model1 = ChatGoogleGenerativeAI(model="gemini-3.8-flash")
model2 = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# 2. Create prompts
prompt1 = PromptTemplate(
    template="Create detailed study notes about:\n{topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Create 5 quiz questions about:\n{topic}",
    input_variables=["topic"],
)

# 3. Initialize the output parser
parser = StrOutputParser()

# 4. Run notes and quiz generation in parallel
parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser,
})

# 5. Create the merging prompt
prompt3 = PromptTemplate(
    template="""
Combine the following notes and quiz into one
well-organized study guide.

NOTES:
{notes}

QUIZ:
{quiz}
""",
    input_variables=["notes", "quiz"],
)

# 6. Merge both outputs
merge_chain = prompt3 | model2 | parser

# 7. Connect the complete workflow
final_chain = parallel_chain | merge_chain

# 8. Execute
result = final_chain.invoke({
    "topic": "Python programming fundamentals"
})

print(result)

final_chain.get_graph().print_ascii()