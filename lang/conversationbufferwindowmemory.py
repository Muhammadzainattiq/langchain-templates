# it is diff from the conversatonbuffermemory bcz it uses a window and trim the old messages.

from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key="AIzaSyA1wygPM_ocs4MgCBTu9DZ3-JCcB9jNelc")
memory = ConversationBufferWindowMemory(k=5) #k is the number of messages
chain = ConversationChain(llm=llm, memory=memory)

while True:
  user_input = input("You:")
  if user_input == "exit":
    break
  response = chain.run(input=user_input)
  print("Bot:", response)