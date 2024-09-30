from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import SimpleSequentialChain

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key="AIzaSyA1wygPM_ocs4MgCBTu9DZ3-JCcB9jNelc")


story_template = """
You have to write a story with the following context:\
{context}
"""

story_prompt_template = PromptTemplate(input_variables=["context"], template=story_template)
story_chain = LLMChain(llm=llm, prompt=story_prompt_template)

translation_template = """
Translate the following story to Urdu:
{story}
"""

translation_prompt_template = PromptTemplate(input_variables=["story"], template=translation_template)
translation_chain = LLMChain(llm=llm, prompt=translation_prompt_template)

improvement_template = """
You have to improve the following story written in urdu. Keep in mind that:\
-It should be well organized.\
-Should maintain the context.\
-The lines should be complete.\
-The story should have right urdu idioms and phrases for native urdu people\
 
Make improvements wherever required in the story and return back a more readable and interesting story for the readers.
Here is the story:\
{story}
"""

improvement_prompt_template = PromptTemplate(input_variables=["story"], template=improvement_template)
improvement_chain = LLMChain(llm=llm, prompt=improvement_prompt_template)

seq_chain = SimpleSequentialChain(chains = [story_chain, translation_chain, improvement_chain])

seq_chain.invoke("A cockroach and lizard")