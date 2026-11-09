from langchain_community.document_loaders import CSVLoader
from dotenv import load_dotenv

load_dotenv()

loader = CSVLoader(
    file_path=r"C:\Users\ayush\OneDrive\Desktop\Langchain\LangChain\data\annual-enterprise-survey-2025-financial-year-provisional-size-bands.csv"
)

documents = loader.load()

print(documents)