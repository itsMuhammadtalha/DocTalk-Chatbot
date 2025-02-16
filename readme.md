# DocTalk - Chat with your personal Documents 

### Live url : https://doctalk-chatbotgit-37jj9xymmvdg7nycpwlpqr.streamlit.app/

### Demo video : https://youtu.be/RU84qcFG9Kk


## Overview
Gemini AI PDF Chat is a Streamlit application that allows users to upload PDF documents and interact with them through a chatbot interface. The application utilizes various libraries, including LangChain and Google Generative AI, to process and analyze the content of the uploaded PDFs.

## Features
- Upload PDF files and extract text.
- Chat with the document using a conversational AI model.
- Visualize token counts from the document.

## Prerequisites
Before running the application, ensure you have Python installed on your machine. This project is compatible with Python 3.7 and above.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Google Generative AI API key. You can do this by modifying the code where the API key is configured.

## Running the Application
To run the Streamlit application, execute the following command in your terminal:



# Code Distribution Overview

This project consists of multiple files that work together to create a Streamlit application for interacting with PDF documents using a chatbot interface. Below is a breakdown of the key files and their responsibilities.

## 1. `app.py`
  - This is the main entry point of the Streamlit application.

  - Sets up the Streamlit UI, including the layout and design.
  - Handles file uploads and processes the uploaded PDF files.
  - Integrates the chatbot functionality, allowing users to ask questions about the document.
  - Displays responses from the AI model based on the content of the PDF.

## 2. `gemini_api.py`
  - Tests the functionality of the Google Generative AI API. before using it 

## 2. `process_pdf.py`
  - Loads PDF files using `PyPDFLoader` and splits them into manageable chunks.
  - Implements a custom embedding class (`GeminiEmbeddings`) to generate embeddings for the text using the Google Generative AI API.
  - Sets up a FAISS vector store to facilitate efficient similarity searches on the document chunks.
  - Defines a conversational retrieval chain that allows users to interact with the document through a chatbot interface.

