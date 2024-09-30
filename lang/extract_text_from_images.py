#%pip install --upgrade --quiet rapidocr-onnxruntime
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("https://arxiv.org/pdf/2103.15348.pdf", extract_images=True)
pages = loader.load()
pages[4].page_content