## Import relevant packages

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
    table_content = table_viewer(temp_file_path)
    st.write(table_content)

    #split the text into tokens so it is not too large for the model to process
    #'\n' uses the new line character as the separator
    text_splitter = CharacterTextSplitter(
        separator="\t",
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
        )

    #pass the content of the table to the splitter to obtain tokens
    tokens = text_splitter.split_text(table_content)

    #create embeddings
    embeddings = OpenAIEmbeddings()

    #create a knowledge base. this knowledge is like a vocabulary
    #it
    knowledge_base = FAISS.from_texts(tokens, embeddings)

    #get the user to give prompts or questions on the pdf
    query = st.text_input("Ask questions about the Tables in the PDF")

    if query is not None:
        #if user inputs a query
        #take the query and search the knowledge base for content related to user quuery
        information = knowledge_base.similarity_search(query)

        #pass the found information to the openai model
        #initialize the model
        model = OpenAI() #here you can specify the type of model

        #load the question answering chain using the openai model
        chain = load_qa_chain(llm=model, chain_type="stuff")

        #run the chain with the information and the user query to generate a response
        response = chain.run(input_documents=information, question=query)

        #display the response generated by the openai model as a success
        st.success(response)