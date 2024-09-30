from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key="AIzaSyA1wygPM_ocs4MgCBTu9DZ3-JCcB9jNelc")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
    MessagesPlaceholder(variable_name="messages"),
])
chain = prompt | model
model_with_message_history = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="messages")

# Creating the prompt template with the MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
    MessagesPlaceholder(variable_name="messages"),
])
for i in range(5):
  human_input = input("Enter your input:")
  messages = [HumanMessage(content=human_input)]

  # Using the prompt template to generate the prompt
  res = model_with_message_history.invoke(

      {"messages": messages},
      config={"configurable": {"session_id": "abc2"}},
  )
  print("messages",  messages)
  print(res.content)
