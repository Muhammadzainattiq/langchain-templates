from itertools import zip_longest
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
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from youtube_transcript_api import YouTubeTranscriptApi

# Configure LangSmith
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "Zain GPT"
# os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

GOOGLE_API_KEY = "AIzaSyA1wygPM_ocs4MgCBTu9DZ3-JCcB9jNelc"



def get_chunks(text):
    text_length = len(text)
    
    # If text length is less than 2000, return the text as a single chunk
    if text_length < 3000:
        return [text]
    
    # If text length is between 2000 and 10000, use chunk size of 3000 and overlap of 600
    elif text_length < 10000:
        chunk_size = 3000
        chunk_overlap = 600
    else:
        # For larger texts, you can define another chunking strategy if needed
        chunk_size = 4000  # Or adjust as required
        chunk_overlap = 800
    
    # Create a text splitter with the determined chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Split the text into chunks
    chunks = text_splitter.split_text(text)
    
    return chunks


def get_vector_store_retriever(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    retriever = vector_store.as_retriever()
    return retriever

def get_chain():
    prompt_template = """
    Your name is "Youtube GPT". You are an assistant to helps me chat with the youtube videos. You will answer all the questions related to a particular video by analyzing the video's transcript given below.\n
    You are provided with three things, the transcript of the video, the user query and the last conversation history. You have to answer the user query using both the transcript of the video and the last conversation history. Use the conversation history history to determine the state of the conversation and answer accordingly.
    keep in mind the following guidelines:
    - If the user greet you, greet him back in a polite manner and try to introduce yourself to them.
    - Your response should be always a plain text string and well organized. Use tables and bullets as per requirement. Avoid giving code cells in the response.
    - Try to answer questions from the transcript of the video as much as possible but if something is mentioned in the transcript of the video video but its details are missing so you can answer them from your knowledge. But don't answer questions unrelated to the transcript of the video.
    - Add at the end of every response: "Any other question about this video."
   \n\n
    Transcript:\n {context}?\n
    User Query: \n{question}\n
    Conversation History: \n{history}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0.8)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "history"])
    chain = load_qa_chain(llm=model, chain_type='stuff', prompt=prompt)
    return chain

def extract_transcript(video_url):
    """Extracts the transcript from a YouTube video."""
    try:
        video_id = video_url.split("=")[1]
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        for transcript in transcript_list:
            transcript_text_list = transcript.fetch()
            lang = transcript.language
            transcript_text = ""
            if transcript.language_code == 'en':
                for line in transcript_text_list:
                    transcript_text += " " + line["text"]
                return transcript_text
            elif transcript.is_translatable:
                english_transcript_list = transcript.translate('en').fetch()
                for line in english_transcript_list:
                    transcript_text += " " + line["text"]
                return transcript_text
        st.info("Transcript extraction failed. Please check the video URL.")
    except Exception as e:
        st.info(f"Error: {e}")

def generate_response(user_question, video_url):
    if "history" not in st.session_state:
        st.session_state["history"] = []
        st.session_state["history"].append(SystemMessage("You are Youtube GPT. You can answer all the queries related to a specific youtube video. And the user will be able chat with the video.")) 



    st.session_state["history"].append(HumanMessage(user_question))
    text = extract_transcript(video_url)
    text_chunks = get_chunks(text)
    retriever = get_vector_store_retriever(text_chunks)
    
    docs = retriever.invoke(user_question)
    
    if not docs: 
        st.write("No relevant context found for the question.")
        return

    chain = get_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question, "history" :st.session_state["history"]}, return_only_outputs=True)
    st.session_state["history"].append(AIMessage(response["output_text"]+"\n"))

    # Separate messages into user and GPT lists, skipping system messages
    user_messages = []
    gpt_messages = []

    for message in reversed(st.session_state["history"]):
        if isinstance(message, SystemMessage):
            continue  # Skip the system message
        elif isinstance(message, HumanMessage):
            user_messages.append(message)
        elif isinstance(message, AIMessage):
            gpt_messages.append(message)

    # Display messages
    for user_message, gpt_message in zip_longest(user_messages, gpt_messages):
        if user_message:
            st.markdown(f"## 👤 <span style='font-size: 17px; font-family: sans-serif;'> {user_message.content}</span>", unsafe_allow_html=True)
            st.markdown("---")
        if gpt_message:
            st.markdown(f"## 🤖 <span style='font-size: 15px; font-family: sans-serif;'> {gpt_message.content}</span>", unsafe_allow_html=True)
            st.markdown("---")



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
    st.set_page_config(page_title="Youtube GPT", page_icon="🎥")
    with st.sidebar: 
     video_url = st.text_input("Enter the video url:")
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

    img_path = "D:\PIAIC\lang\lang\chat_with_vid\yt.png"  # Ensure this path points to a valid image file
    img_base64 = img_to_base64(img_path)
    if img_base64:
        st.sidebar.markdown("""
            <div class="sidebar-title">Youtube GPT</div>
            <img src="data:image/png;base64,{img_base64}" class="cover-glow">
        """.format(img_base64=img_base64), unsafe_allow_html=True)
    else:
        st.sidebar.write("Image not found or error loading image.")
    
    st.sidebar.markdown("""
    <div class="sidebar-section">
        <div class="section-title">About Youtube GPT</div>
        <div class="section-description">
            Youtube GPT is a cutting-edge AI application that allows you to ask questions about any YouTube video. The app also maintains conversation history, creating a friendly and interactive environment. Follow the steps below to get started:
            <ol>
                <li>Visit <a href="https://youtube.com" target="_blank">YouTube</a>.</li>
                <li>Copy the URL of the video you want to explore.</li>
                <li>Paste the video URL in the sidebar and press Enter.</li>
                <li>Enter your question in the text box provided.</li>
                <li>Press Enter and wait a moment for your answer to appear.</li>
            </ol>
            Feel free to ask anything related to the video and enjoy a seamless experience!
        </div>
    </div>
""", unsafe_allow_html=True)


  

    st.markdown('<div class="custom-header">Youtube GPT 🎥 🤖 </div>', unsafe_allow_html=True)


    # # Process the PDF and create embeddings once
    # if not os.path.exists("faiss_index"):
    #     with st.spinner("Processing PDF and creating embeddings..."):
    #         text = extract_transcript(video_url)
    #         if text:
    #             text_chunks = get_chunks(text)
    #             if text_chunks:
    #                 get_vector_store_retriever(text_chunks)
    #                 st.success("Embeddings created!")
    #             else:
    #                 st.error("Couldn't find video transcript")

    #         else:
    #             st.error("Couldn't find video transcript")
            
            

    user_question = st.text_input("Ask a question from video:")
    if video_url:
        if user_question:
            with st.spinner("Generating Answer...Please Wait..."):
                generate_response(user_question = user_question, video_url=video_url)
        else:
            st.warning("Please enter a question.")

    else:
        st.warning("please enter the video url first to chat.")

if __name__ == "__main__":
    main()
