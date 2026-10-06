from langchain_core.prompts import PromptTemplate

template=PromptTemplate(
    input_variables=["movie", "size", "style"],
    template="Write a {size} paragraph summary of the research paper {movie} in a {style} style."
)

template.save("template.json")