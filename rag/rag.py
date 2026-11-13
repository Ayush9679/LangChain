from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from dotenv import load_dotenv
load_dotenv()

video_id = "J5_-l7WIO_w"

try:
    ytt_api = YouTubeTranscriptApi()

    transcript = ytt_api.fetch(
        video_id,
        languages=["hi"]
    )

    transcript_text = " ".join(
        [snippet.text for snippet in transcript]
    )

    print(transcript_text)

except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")


splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,)

split_text=splitter.create_documents([transcript_text])


embedding=GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001")

vectorstore=FAISS.from_documents(split_text,embedding)

retriever=vectorstore.as_retriever(search_kwargs={"k":3})

model=ChatGoogleGenerativeAI(
    model="gemini-3.7-flash")

prompt=PromptTemplate(
    input_variables=["context","question"],
    template="Answer the question based on the context below. If the context is not sufficient, say 'I don't know'.\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
)
question=input("Enter your question: ")
context=retriever.invoke(question)

chain=prompt | model

result=chain.invoke({"context":context,"question":question})

print(result)
