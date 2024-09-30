import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

# Configure Google Generative AI
# genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
genai.configure(api_key="AIzaSyA1wygPM_ocs4MgCBTu9DZ3-JCcB9jNelc")


# Function to extract text from a PDF file
def get_pdf_text(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text = ""
    for doc in documents:
        text += doc.page_content
    return text
separators = ["Personal Information", "Education", "Technical Skills", "Degrees/Certifications/Courses:", "Projects :", "Experience:"]
# Function to split the extracted text into smaller chunks for embedding
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50, separators=separators)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store and save embeddings using Google Generative AI
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key="AIzaSyA1wygPM_ocs4MgCBTu9DZ3-JCcB9jNelc")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local('faiss_index')

# Function to create a question-answering chain using Google Generative AI
def get_chain():
    prompt_template = """
    You are Zain GPT. You are here to answer all the question related to Zain Attiq who is a Cloud Applied Generative AI Engineer.\n
    Answer any question about zain and anything related to him as detailed as possible from the provided context. If answer is not availabe in the provided context. Just say: "Sorry I Can't answer this right now. You can contact Zain Attiq at zainatteeq@gmail.com." Never provide a wrong answer.\n
    Keep the following things in consideration:\n
    - Your answer should be well organized using bullet points and tables whereever required and must add a starting line by yourself before the real information to look more realistic and human like.
    - If the question is about Zain Attiq try to answer it from the context provided in a polite and sweet manner.
    - If it is a simple greeting. Greet back in a polite and sweet way. Introduce yourself as "Zain GPT" and ask them that what they want to know about Zain Attiq.
    - You are only developed to answer questions related to Zain. If the question is something unrelated to Zain Attiq, Never answer and tell back that I can only answer questions related to Zain Attiq 
    -If the context have project links and video presentations link in it, must include them in the response.
   \n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key="AIzaSyA1wygPM_ocs4MgCBTu9DZ3-JCcB9jNelc", temperature=0.4)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type='stuff', prompt=prompt)
    return chain

# Function to generate a response based on a user's question and the document embeddings
def generate_response(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key="AIzaSyA1wygPM_ocs4MgCBTu9DZ3-JCcB9jNelc")
    new_db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
    
    # Print the retrieved docs to verify the output
    docs = new_db.similarity_search(user_question)
    st.write("Retrieved Docs:", docs)  # Check if correct docs are being retrieved
    
    if not docs:
        st.write("No relevant context found for the question.")
        return

    chain = get_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:\n", response["output_text"])


# Main function to run the Streamlit app
def main():
    st.set_page_config("Zain GPT", page_icon=":card_index_dividers:")

    st.markdown("""
        <style>
            .header {
                font-size: 46px;
                color: #1E90FF;
                text-align: center;
                margin-bottom: 30px;
            }
            .footer-container {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-top: 50px;
            }
            .footer-text {
                color: #888888;
                font-size: 22px;
            }
            .link-container {
                text-align: right;
            }
            .link {
                display: inline-block;
                margin: 0 10px;
                padding: 5px 7px;
                background-color: #f4f4f4;
                color: #333;
                text-decoration: none;
                font-weight: bold;
                border-radius: 5px;
                font-size: 12px;
                transition: background-color 0.3s ease, color 0.3s ease;
            }
            .link:hover {
                background-color: #0073b1;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="footer-container">
            <div class="footer-text">➡️ created by Muhammad Zain Attiq</div>
            <div class="link-container">
                <a class="link" href="mailto:zainatteeq@gmail.com" target="_blank">Email</a>
                <a class="link" href="https://www.linkedin.com/in/muhammadzainattiq/" target="_blank">LinkedIn</a>
                <a class="link" href="https://github.com/Muhammadzainattiq" target="_blank">GitHub</a>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.header("Zain GPT🤖📄")

    # Static file path for the PDF to be processed (change to the actual path on your backend)
    pdf_path = "D:\PIAIC\lang\lang\zaingptpdf.pdf"  # Replace with your actual path

    # Process the PDF and create embeddings once
    if not os.path.exists("faiss_index"):
        with st.spinner("Processing PDF and creating embeddings..."):
            raw_text = get_pdf_text(pdf_path)
            if raw_text:
                text_chunks = get_chunks(raw_text)
                st.write("test_chunks.txt", text_chunks)
                if text_chunks:
                    get_vector_store(text_chunks)
                    st.success("Embeddings created!")
            else:
                st.error("Couldn't extract text from the PDF.")

    # Input box for the user's question
    question = st.text_input("Enter your question:", value="What is in this PDF?")
    if question:
        try:
            generate_response(question)
        except Exception as e:
            st.error(e)

if __name__ == "__main__":
    main()
