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
import base64
import logging
from langchain.memory import ChatMessageHistory
# Configure Google Generative AI
genai.configure(api_key="AIzaSyA1wygPM_ocs4MgCBTu9DZ3-JCcB9jNelc")

def get_pdf_text(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text = "".join(doc.page_content for doc in documents)
    st.write("extracted text", text)
    return text

separators = ["Personal Information", "Education", "Technical Skills", "Degrees/Certifications/Courses:", "Projects :", "Experience:"]

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50, separators=separators)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key="AIzaSyA1wygPM_ocs4MgCBTu9DZ3-JCcB9jNelc")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local('faiss_index')

def get_chain():
    prompt_template = """
    You are Zain GPT. You are here to answer all the questions related to Zain Attiq who is a Cloud Applied Generative AI Engineer. Provide organized and detailed human like responses to the queries related to Zain by using the Context provided below and Conversation History. Must pay attention at the Coversation history to understand well that what is the state of the conversation and try to carry it on.\n
    \n
    Keep the following things in consideration:\n
    - Answer to the greetings in a polite and sweet manner and introduce yourself.
    - Your answer should be well organized using bullet points and tables where necessary and must add a starting line by yourself before the real information to look more human-like in a conversational way.
    - If the user has asked any question  about Zain Attiq, try to answer it in much detailed way in a polite and sweet manner.
    - You are only developed to answer questions related to Zain. If the question is something unrelated to Zain Attiq, never answer and tell them that you can only answer questions related to Zain Attiq.
    - If the user asked about the projects must provide them with the project names with the respective project links and video presentation links from the context.
    - If answer of the user query is not available in the provided context, just say: "Sorry I can't answer this right now. You can contact Zain Attiq at zainatteeq@gmail.com." Never provide a wrong answer.
   \n\n
    Context:\n {context}?\n
    User Query: \n{question}\n
    Conversation History: \n{history}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key="AIzaSyA1wygPM_ocs4MgCBTu9DZ3-JCcB9jNelc", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "history"])
    chain = load_qa_chain(llm=model, chain_type='stuff', prompt=prompt)
    return chain

def generate_response(user_question):
    if "history" not in st.session_state:
        st.session_state["history"] = ChatMessageHistory()

    st.write("history before user_question", st.session_state["history"])


    st.session_state["history"].add_user_message(user_question +"\n")
    st.write("history after adding user message: " ,st.session_state["history"])
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key="AIzaSyA1wygPM_ocs4MgCBTu9DZ3-JCcB9jNelc")
    new_db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)
    
    if not docs:
        st.write("No relevant context found for the question.")
        return

    chain = get_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question, "history" :st.session_state["history"]}, return_only_outputs=True)
    st.session_state["history"].add_ai_message(response["output_text"]+"\n")
    st.write("history after adding ai message: " ,st.session_state["history"])
    st.write("Zain GPT:\n", response["output_text"])

def img_to_base64(image_path):
    try:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            return base64.b64encode(img_data).decode()
    except FileNotFoundError as e:
        logging.error(f"File not found: {str(e)}")
    except Exception as e:
        logging.error(f"Error converting image to base64: {str(e)}")
    return None

def main():
    st.set_page_config(page_title="Zain GPT", page_icon=":card_index_dividers:")

    st.markdown("""
        <style>
            .custom-header {
                font-size: 40px;
                color: #FF0000; /* Red color */
                text-align: center;
                font-weight: bold;
                background-color: #000000; /* Black background */
                border: 5px solid #F93822; /* Red border */
                padding: 2px;
                border-radius: 2px;
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.6); /* Black shadow for depth */
                margin: 5px 0;
                text-shadow: 3px 3px 6px #000000, 0 0 25px #FF0000, 0 0 5px #FF0000;
            }
            .footer-container {
                display: flex;
                justify-content: space-between;
                align-items: center;
                position: fixed;
                width: 100%;
                bottom: 0;
                background-color: #000000;
                color: #FFD700;
                padding: 10px;
                box-shadow: 0px -2px 10px rgba(0, 0, 0, 0.5);
                z-index: 10;
            }
            .link-container {
                text-align: right;
            }
            .link {
                display: inline-block;
                margin: 0 10px;
                padding: 5px 7px;
                background-color: #000000;
                color: #FFD700;
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
            .sidebar-title {
                font-size: 44px;
                color: #FF0000;
                text-align: center;
                vertical-align: top;
                font-weight: bold;
                padding-bottom: 20px;
                text-shadow: 3px 3px 6px #000000, 0 0 25px #FF0000, 0 0 5px #FF0000;
            }
            .sidebar-section {
                background-color: #000000;
                border-radius: 10px;
                padding: 15px;
                margin-top: 20px;
            }
            .section-title {
                font-size: 18px;
                font-weight: bold;
                color: #FF0000;
                margin-bottom: 10px;
            }
            .section-description {
                font-size: 14px;
                color: #FFFFFF;
                line-height: 1.6;
            }
                .cover-glow {
                width: 100%;
                height: auto;
                padding: 3px;
                box-shadow: 
                    0 0 5px #330000,
                    0 0 10px #660000,
                    0 0 15px #990000,
                    0 0 20px #CC0000,
                    0 0 25px #FF0000,
                    0 0 30px #FF3333,
                    0 0 35px #FF6666;
                position: relative;
                z-index: -1;
                border-radius: 45px;
        </style>
    """, unsafe_allow_html=True)

    img_path = "D:/PIAIC/lang/lang/zain.jpg"  # Ensure this path points to a valid image file
    img_base64 = img_to_base64(img_path)
    if img_base64:
        st.sidebar.markdown("""
            <div class="sidebar-title">Zain GPT</div>
            <img src="data:image/png;base64,{img_base64}" class="cover-glow">
        """.format(img_base64=img_base64), unsafe_allow_html=True)
    else:
        st.sidebar.write("Image not found or error loading image.")
    
    st.sidebar.markdown("""
    <div class="sidebar-section">
        <div class="section-title">About Zain GPT</div>
        <div class="section-description">
            Zain GPT is an advanced GPT designed to answer questions related to Zain Attiq, a Cloud Applied Generative AI Engineer. Feel free to ask about Zain's:
            <ul>
                <li>Personal Life</li>
                <li>Professional Life</li>
                <li>Social Life</li>
                <li>Liking and Disliking</li>
                <li>Projects</li>
                <li>Skills</li>
                <li>Experiences</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="footer-container">
            <div class="footer-text">👉  Created by Muhammad Zain Attiq</div>
            <div class="link-container">
                <a class="link" href="mailto:zainatteeq@gmail.com" target="_blank">Email</a>
                <a class="link" href="https://www.linkedin.com/in/muhammadzainattiq/" target="_blank">LinkedIn</a>
                <a class="link" href="https://github.com/Muhammadzainattiq" target="_blank">GitHub</a>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-header">Zain GPT🤖👨🏻‍💻</div>', unsafe_allow_html=True)

    user_question = st.text_input("Ask a question about Zain Attiq:")

    if user_question:
        with st.spinner("Generating Answer...Please Wait..."):
            generate_response(user_question)
    else:
        st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
