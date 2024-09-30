#it will create the summary of the conversation.

from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key="AIzaSyA1wygPM_ocs4MgCBTu9DZ3-JCcB9jNelc")
memory = ConversationSummaryMemory(llm = llm) #it will use the llm to generate a summary of the previous conversation
chain = ConversationChain(llm=llm, memory=memory)

while True:
  user_input = input("You:")
  if user_input == "exit":
    break
  response = chain.run(input=user_input)
  print("Bot:", response)