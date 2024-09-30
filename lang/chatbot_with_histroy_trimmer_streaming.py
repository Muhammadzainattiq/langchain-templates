from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import trim_messages
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

store = {}
messages = []

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key="AIzaSyA1wygPM_ocs4MgCBTu9DZ3-JCcB9jNelc")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI professor. Whenever someone ask you: who are you? tell them that you are a AI professor. "),
    MessagesPlaceholder(variable_name="messages")
])

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)


chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)
model_with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

config = {"configurable": {"session_id": "abc20"}}

for i in range(10):
    human_input = input("Enter your input: ")
    human_message = HumanMessage(content=human_input)
    messages.append(human_message)
    trimmer.invoke(messages)
    # Filter out the SystemMessage before passing to the model
    filtered_messages = [msg for msg in messages if isinstance(msg, (HumanMessage, AIMessage))]

    # Stream the AI response
    print("AI: ", end="")
    ai_response_content = ""  # To accumulate the AI message content
    for r in model_with_message_history.stream({"messages": filtered_messages}, config=config):
        print(r.content, end="")  # Print each token as it's generated
        ai_response_content += r.content  # Accumulate the tokens
    print()  # Newline after the response is complete

    # Create the full AIMessage object
    ai_message = AIMessage(content=ai_response_content)
    messages.append(ai_message)  # Add the final response to the message history
    print("messages", messages)
