
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-3.8-flash")

import streamlit as st

user_input=st.text_input("Enter the prompt")

if st.button("Submit"):
    response=model.invoke(user_input)
    st.write(response.text)

