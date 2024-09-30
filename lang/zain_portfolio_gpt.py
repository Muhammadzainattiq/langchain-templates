import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyA1wygPM_ocs4MgCBTu9DZ3-JCcB9jNelc"
# Set page configuration
st.set_page_config(page_title="Zain's Portfolio GPT", page_icon=":robot:")

# Title and description
st.title("Zain's Portfolio GPT")
st.write("Ask questions about Zain's portfolio and get answers!")

# Load portfolio file
portfolio_file = "D:\PIAIC\lang\lang\portfolio_gpt.txt"

# Load documents
loader = TextLoader(portfolio_file)
documents = loader.load()

# Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_query")

# Create vector store
db = Chroma.from_documents(documents, embeddings)

# Create retriever
retriever = db.as_retriever()

# Create question answering chain
qa = RetrievalQA.from_chain_type(
    llm=ChatGoogleGenerativeAI(model_name="gemini-1.5-flash", google_api_key= "AIzaSyA1wygPM_ocs4MgCBTu9DZ3-JCcB9jNelc"),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# User input
try:
    query = st.text_input("Ask a question about Zain's portfolio:")
    if query:
        result = qa.run(query)
        st.write(result)

        st.write("Source documents:")
        if "result" in result and "source_documents" in result["result"]:
            for doc in result["result"].source_documents:
                st.write(doc.page_content)

except Exception as e:
    st.error(f"An error occurred: {e}")
