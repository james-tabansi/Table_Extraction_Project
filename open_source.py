## Import relevant packages

#for computation
import numpy as np

#for building user interface
import streamlit as st
#for making a temporary folder to store pdf file
import os
import tempfile

#for reading tables inside the pdf
import fitz
from pprint import pprint

#use the langchain framework to build apps powered by LLM
#to split the texts inside the pdf into smaller chunks
from langchain.text_splitter import CharacterTextSplitter

#load the model
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI, HuggingFaceHub

#for loading environment variable
from dotenv import load_dotenv

#use the langchain framework to build apps powered by LLM
#to get the word-embeddings for the tokens using openai embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

#to create vectors from the embeddings
from langchain.vectorstores import FAISS

from transformers import T5Tokenizer, T5ForConditionalGeneration

#load the environment variable
load_dotenv(dotenv_path='.env')

#load the huggingface key
hugging_face_api_key = os.getenv("HUGGING_FACE_API_KEY")

#embeddings = HuggingFaceEmbeddings()



#welcome user
st.write("## Welcome to Table Extraction Web Application using TAPAS")


#allow pdf file upload
pdf = st.file_uploader("Upload your pdf", type='pdf')

from functions import get_table_objects, convert_table_to_df, prompter, TapasQAprompter
try:
        #get the table objects
        table_objects, ind  = get_table_objects(pdf)


        page_num = st.sidebar.selectbox("Select page to view Table content", options=ind) - 1

        #display the selected table
        st.table(table_objects[int(page_num)].extract())
        #convert table to string
        content = convert_table_to_df(table_objects, int(page_num))
        #st.write(content)
        #get user's query
        query = st.text_input("Ask questions about the Tables in the PDF")

        #create prompt, pass it to llm, and get a response
        #answer = prompter(query=query, content=content)

        answer = TapasQAprompter(query=query, content=content)


        st.write(answer)
except TypeError:
        pass


