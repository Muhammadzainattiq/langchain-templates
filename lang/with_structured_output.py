#To get the answer in a structured format we can use with_structured_ouput functionality of chatmodels. availabel also for gemini models.

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

class Person(BaseModel):
  name: str = Field(description = "The name of the person")
  gender: str = Field(description = "The gender of the person")
  age: int = Field(description = "The age of the person")
  job: str = Field(description = "The job title of the person")
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key = "AIzaSyA1wygPM_ocs4MgCBTu9DZ3-JCcB9jNelc")
# ans = model.invoke("HI! what you are busy doing?")
structured_model = model.with_structured_output(Person)
structured_model.invoke("hi! I'm zain and m a boy of 18 and i work as a software engineer.")