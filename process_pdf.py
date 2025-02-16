import pypdf
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import textract
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
import numpy as np




loader = PyPDFLoader("chatbot.pdf")
pages = loader.load_and_split()
print(pages[0])

chunks = pages 

# converting pdf to text

doc = textract.process("chatbot.pdf")

# Save to .txt and reopen (helps prevent issues)
with open('demo_pdf.txt', 'w') as f:
    f.write(doc.decode('utf-8'))

with open('demo_pdf.txt', 'r') as f:
    text = f.read()


# now convert to chuncks 

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 512,
    chunk_overlap  = 24,
    
)    


chunks = text_splitter.create_documents([text])

# Configure Gemini API Key
genai.configure(api_key="AIzaSyCAo41zFdx6v__JvIRZIPo18Zvpa60c_Dg")

# Initialize the Gemini Pro model
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


# Define Query
query = "What are UI requirements?"

# Retrieve relevant documents from FAISS
docs = db.similarity_search(query)
context = "\n\n".join([doc.page_content for doc in docs])

# Use Gemini to generate an answer
response = model.generate_content(f"Context: {context}\n\nQuestion: {query}\n\nAnswer:")
print(response.text)