from http.client import responses

import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from streamlit.navigation import page
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI

from openai import OpenAI

OPEN_API_KEY=""

#Upload PDF files

st.header("My First Chatbot")
with st.sidebar:
    st.title("Your Document")
    file=st.file_uploader("Upload a file and start asking questions", type="pdf")

#Extract the text
if file is not None:
    pdf_reader=PdfReader(file)
    text=""
    for page in pdf_reader.pages:
        text+=page.extract_text()
        #st.write(text)

    #Break it into chunks
    text_splitter=RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )

    chunks=text_splitter.split_text(text)
    #st.write(chunks)

    #Generating embeddings
    embeddings=OpenAIEmbeddings(openai_api_key=OPEN_API_KEY)

    #Creating vector store =FAISS
    vector_store=FAISS.from_texts(chunks,embeddings)

    #Get user question
    user_question=st.text_input("Type your question here")

    #Do similarity search
    if user_question:
        match=vector_store.similarity_search(user_question)
        #st.write(match)

        #Define the LLM
        llm=ChatOpenAI(
            openai_api_key=OPEN_API_KEY,
            temperature=0,
            max_tokens=100,
            model_name="gpt-4o-mini",
        )

        #Output result
        chain=load_qa_chain(llm,chain_type="stuff")
        chain.run(input_documents = match,question=user_question)
        st.write(responses)


