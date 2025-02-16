from process_pdf import chunks
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
import numpy as np

# Gemini API Key (add your mine will be removed shorter )
genai.configure(api_key="AIzaSyCAo41zFdx6v__JvIRZIPo18Zvpa60c_Dg") 

# model selection
model = genai.GenerativeModel("gemini-pro")

# Define a custom embedding function for FAISS
class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text):
        return self._embed_text(text)

    def _embed_text(self, text):
        response = genai.embed_content(model="models/text-embedding-004", content=text, task_type="retrieval_document")
        return response["embedding"]

# Initialize Gemini embeddings
embeddings = GeminiEmbeddings()

# Create FAISS vector store
db = FAISS.from_documents(chunks, embeddings)
