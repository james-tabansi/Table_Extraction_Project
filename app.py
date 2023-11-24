## Import relevant packages

#for building user interface
import streamlit as st
#for making a temporary folder to store pdf file
import os
import tempfile

#for reading tables inside the pdf
import fitz
from pprint import pprint

#load the model
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

#for loading environment variable
from dotenv import load_dotenv

#use the langchain framework to build apps powered by LLM
#to get the word-embeddings for the tokens using openai embeddings
from langchain.embeddings.openai import OpenAIEmbeddings

#to create vectors from the embeddings
from langchain.vectorstores import FAISS

#load the environment variable
load_dotenv(dotenv_path='.env')

## design the user interface
#set the title
st.title("Table Extraction Application")
#display message to user
st.header("Upload your PDF to read the tables")

#allow pdf file upload
pdf = st.file_uploader("Upload your pdf", type='pdf')

from utility_fns import table_viewer

#if pdf is uploaded
if pdf is not None:
    #get the name of the pdf file
    name = pdf.name
    #display the name
    # st.write(name)
    #create a temporal file path to locate the pdf
    temp_file_path = os.path.join(tempfile.mkdtemp(), str(name))

    #save the pdf file inside a temporal folder such that the file path will now exist
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(pdf.read())

    #display the tables found in the file with accurate description of the pages found and the table number
    tables = table_viewer(temp_file_path)
