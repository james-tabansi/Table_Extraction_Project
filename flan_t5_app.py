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
from langchain import HuggingFaceHub


#for loading environment variable
from dotenv import load_dotenv
huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

#use the langchain framework to build apps powered by LLM
#to get the word-embeddings for the tokens using openai embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings #for using HugginFace models


#to create vectors from the embeddings
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator #vectorize db index with chromadb

#for question answering chain
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain

#for prompt engineering
from langchain.prompts import PromptTemplate

#load the environment variable
load_dotenv(dotenv_path='.env')

from functions import get_table_objects, convert_table_to_string

#instantiate the llm
llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.7, "max_length":5000})

#instantiate the huggingface embeddings
embeddings = HuggingFaceEmbeddings()


#instantiate the question-answering chain
chain = load_qa_chain(llm, chain_type="stuff")


#welcome user
st.write("## Welcome to Table Extraction Web Application Using FLAN-T5")


#allow pdf file upload
pdf = st.file_uploader("Upload your pdf", type='pdf')

try:
    #get the table objects
    table_objects, ind  = get_table_objects(pdf)


    page_num = st.sidebar.selectbox("Select page to view Table content", options=ind) - 1

    #display the selected table
    st.table(table_objects[int(page_num)].extract())
    #convert table to string
    content = convert_table_to_string(table_objects, int(page_num))

    #get user's query
    query = st.text_input("Ask questions about the Tables in the PDF")

    #Prompt Engineering
    p_temp = f"""You will be given as {content} the content of a table as string and required to answer {query}
    or perform mathematical computations using the values in the table
    Example: input: Term Undergraduate Graduate 0 Fall 2019 19886 3441 1 Winter 2020 19660 3499 2 Spring 2020 19593 3520, 
    question: what is the sum of the values in the second column; answer:19886+19660+19593=59139"
    """
    query_template = PromptTemplate(input_variables=['content', 'question'], template=p_temp)



    # Define chunk size, overlap and separators
    text_splitter = CharacterTextSplitter(separator="\n",chunk_size=1000,
                                    chunk_overlap=200,
                                    length_function=len)
    #split the content of the selected table
    tokens = text_splitter.split_text(p_temp)

    #create a vector database to hosts the information
    db = FAISS.from_texts(tokens, embeddings)


    #perform similarity search using the user's query
    docs = db.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)

    #display the response generated by the Flan-T5 model as a success
    st.success(response)


    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}))


except TypeError:
        pass


