
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt
load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-3.6-flash")

import streamlit as st

movie_input=st.selectbox("Select the Research paper", ["Harry Potter", "Fantastic Beasts","Mirzapur"], index=1)
size_input=st.selectbox("Select the No of paragraphs", ["1", "2", "3"], index=1)
style_input=st.selectbox("Select the style of writing", ["Formal", "Informal","Humorous"], index=1)

template=load_prompt("template.json")

if st.button("Submit"):
    chain=template | model
    response=chain.invoke(
        {
            "movie":movie_input,"size":size_input,"style":style_input
        }
    )
    st.write(response.text)

