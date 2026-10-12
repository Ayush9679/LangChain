from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal

load_dotenv()

class review(TypedDict):
    key_terms: Annotated[list[str], "A list of key terms extracted from the review"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["positive", "negative", "neutral"], "The sentiment of the review"]
    pros: Annotated[list[str], "A list of positive aspects mentioned in the review"]
    cons: Annotated[list[str], "A list of negative aspects mentioned in the review"]
    names:Annotated[str, "The name of the reviewer"]
model=ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)

structured_model=model.with_structured_output(review)

result=structured_model.invoke("""The Samsung Galaxy S24 Ultra is a premium flagship smartphone with excellent overall performance.  
Its Snapdragon 8 Gen 3 processor handles gaming, multitasking, and demanding apps smoothly.  
The 6.8-inch Dynamic AMOLED 2X display is bright, sharp, and supports a smooth 120Hz refresh rate.  
The 200MP main camera captures highly detailed photos with strong image processing.  
Its 3x and 5x optical zoom cameras make it excellent for capturing distant subjects.  
The built-in S Pen is useful for taking notes, drawing, editing, and productivity tasks.  
Galaxy AI features add useful tools such as Circle to Search, Live Translate, and Photo Assist.  
The 5000mAh battery provides strong battery life for everyday use and heavy usage.
The titanium build gives the phone a premium and durable feel.  
**Pros:** Excellent display, powerful performance, versatile cameras, S Pen, strong battery life, and useful AI features.  
**Cons:** The phone is quite heavy, expensive, has no expandable storage, and lacks a 3.5mm headphone jack.
Overall, the S24 Ultra is an excellent choice for users who want a powerful flagship with a great camera, display, and productivity features.
Reviewed by Ayush Kumar Dubey""")

print(result["sentiment"])
print(result["names"])