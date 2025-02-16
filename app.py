import streamlit as st
import google.generativeai as genai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
import os


# ------------------- UI DESIGN -------------------
st.set_page_config(page_title="DocTalk : AI Pdf Chat", page_icon="📄", layout="centered")

st.markdown("""
    <style>
        .stApp {
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
            background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #f9a826;
            margin-bottom: 2rem;
        }
        .stTextInput > div > div > input {
            border-radius: 20px;
            background-color: rgba(255,255,255,0.1);
            border: none;
            color: white;
            padding: 0.5rem 1rem;
        }
        .stFileUploader > div > div > button {
            border-radius: 20px;
            background-color: #f9a826;
            color: #0f2027;
            border: none;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
        }
        .stFileUploader > div > div > button:hover {
            background-color: #ffc107;
        }
        .stButton > button {
            border-radius: 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 0.5rem 2rem;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .stAlert {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("DocTalk 📄 : Chat with your Documents")

# ------------------- PDF UPLOAD -------------------
uploaded_file = st.file_uploader("📤 Upload a PDF file to chat with it:", type=["pdf"])

if uploaded_file:
    st.success("✅ File uploaded successfully! Processing...")

    # Save PDF
    pdf_path = "temp.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # ------------------- TEXT EXTRACTION -------------------

    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    text = "\n\n".join([page.page_content for page in pages])

    # ------------------- TEXT SPLITTING -------------------

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=24)
    chunks = text_splitter.create_documents([text])

    # ------------------- EMBEDDING & FAISS VECTOR STORE -------------------
    
    genai.configure(api_key="AIzaSyCAo41zFdx6v__JvIRZIPo18Zvpa60c_Dg") 

    class GeminiEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [self._embed_text(text) for text in texts]

        def embed_query(self, text):
            return self._embed_text(text)

        def _embed_text(self, text):
            response = genai.embed_content(model="models/text-embedding-004", content=text, task_type="retrieval_document")
            return response["embedding"]

    embeddings = GeminiEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)

    # ------------------- CHATBOT INTERFACE -------------------

    st.subheader("💬 Chat with your document!")

    query = st.text_input("Ask something about the document:")

    if st.button("Get Answer"):
        if query:
            with st.spinner("🤖 Thinking..."):
                docs = db.similarity_search(query)
                context = "\n\n".join([doc.page_content for doc in docs])

                model = genai.GenerativeModel("gemini-pro")
                response = model.generate_content(f"Context: {context}\n\nQuestion: {query}\n\nAnswer:")

                st.success("✅ Response Generated!")
                st.write("🧠 **Answer:**")
                st.write(response.text)
        else:
            st.warning("⚠ Please enter a question!")

