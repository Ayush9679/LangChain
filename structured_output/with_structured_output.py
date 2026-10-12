from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal

load_dotenv()

json_schema={
  "type": "object",
  "properties": {
    "key_terms": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "A list of key terms extracted from the review"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": [
        "positive",
        "negative",
        "neutral"
      ],
      "description": "The sentiment of the review"
    },
    "pros": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "A list of positive aspects mentioned in the review"
    },
    "cons": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "A list of negative aspects mentioned in the review"
    },
    "names": {
      "type": "string",
      "description": "The name of the reviewer"
    }
  },
  "required": [
    "key_terms",
    "summary",
    "sentiment",
    "pros",
    "cons",
    "names"
  ]
}

model=ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2)

struc_model=model.with_structured_output(json_schema)

response=struc_model.invoke("""The Samsung Galaxy S24 Ultra is a premium flagship smartphone with excellent overall performance.  
Its Snapdragon 8 Gen 3 processor handles gaming, multitasking, and demanding apps smoothly.""")

print(response)