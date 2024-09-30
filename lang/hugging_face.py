import os
from langchain import HuggingFaceHub

# Set your HuggingFace API token
os.environ["HUGGINGFACE_TOKEN"] = "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"

# Load the HuggingFace API token from environment variables
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Initialize the HuggingFace model
llm3 = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2", huggingfacehub_api_token=HUGGINGFACE_TOKEN)

# Define your query
text = "What is the capital city of Pakistan and what is famous about it?"

# Get the response from the model
response = llm3.invoke(text)
print(response)