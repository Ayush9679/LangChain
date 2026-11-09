import os
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

os.environ.setdefault("USER_AGENT", "AyushLangChainLearning/1.0")

model = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash"
)

url = "https://www.flipkart.com/samsung-galaxy-book4-metal-intel-core-i5-13th-gen-1335u-16-gb-512-gb-ssd-windows-11-home-np750xgj-thin-light-laptop/p/itm1df5212efb9fc"

loader = WebBaseLoader(web_path=url)

docs = loader.load()

prompt = PromptTemplate(
    template="""
    Answer the question using only the following webpage content.

    Question: {question}

    Webpage content:
    {text}

    If the answer is not available in the content, say so.
    """,
    input_variables=["question", "text"]
)

chain = prompt | model

result = chain.invoke({
    "question": "What is the price of the laptop?",
    "text": docs[0].page_content
})

print(result.content)