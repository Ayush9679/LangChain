from pydantic import BaseModel,Field
from typing import Optional 
class Student(BaseModel):
    name: str
    age: Optional[int] = None
    cgpa: float=Field(gt=0, lt=10, description="CGPA must be between 0 and 10")

new_student = {"name":"Ayush Kumar Dubey", "age": 22, "cgpa": 9.9}

student=Student(**new_student)

response=dict(student)
print(response)




