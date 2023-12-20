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

#welcome user
st.write("## Welcome to Table Extraction Web Application")


#allow pdf file upload
pdf = st.file_uploader("Upload your pdf", type='pdf')

from utility_fns import (pdf_reader, page_finder, table_finder, single_table_disp, multi_table_disp, 
table_2_llm, query_llm)

#if pdf is uploaded
if pdf is not None:
    #get the name of the pdf file
    name = pdf.name

    #create a temporal file path to locate the pdf
    temp_file_path = os.path.join(tempfile.mkdtemp(), str(name))

    #save the pdf file inside a temporal folder such that the file path will now exist
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(pdf.read())
    
    doc = pdf_reader(temp_file_path)

    #get the pages where tables are found for easier user selection
    pages = page_finder(doc)

    #display an appropriate message to user
    # if len(pages) > 1:
    #     st.write(f"There are {len(pages)} pages where at least one table is found")

    # elif len(pages) == 1:
    #     st.write(f"Table is found in a single page")

    # else:
    #     st.write(f"No page in PDF contain a table")

    #get the number of tables in the pdf
    count = table_finder(doc)
    #display an appropriate message for user
    #st.write(f"There are {count} tables in this pdf")

    #provided the document has been read and there is at least 1 table in the PDF
    if doc is not None and count > 0:
        # #create a sidebar with the header
        #st.sidebar.subheader('Select Choice of Display')
        # #ask user to select display option
        # value = st.sidebar.radio(label="Select display option", options=['Single Table', 'Range of Tables'])

        # #if user selects Single Table
        # if value == 'Single Table':
        #get the page number
        page_num = st.sidebar.selectbox("Select page to view Table content", options=np.array(pages))
        but1 = st.sidebar.button("Click to view Table")
        try:
            if but1:
                #display the table(s) in that page alone
                table = single_table_disp(doc=doc,page_num=int(page_num))

                #preprocess the table
                content = table_2_llm(table)
                    
                #split the content into tokens so it is not too large for the model to process
                #'\n' uses the new line character as the separator
                text_splitter = CharacterTextSplitter(separator="\n",chunk_size=1000, 
                            chunk_overlap=200,
                            length_function=len)
                st.write(content)
                #pass the content of the table to the splitter to obtain tokens
                tokens = text_splitter.split_text(content)

                #create embeddings
                embeddings = OpenAIEmbeddings()

                #create a knowledge base. this knowledge is like a vocabulary
                #it
                knowledge_base = FAISS.from_texts(tokens, embeddings)

                # #ask the user if he wants to ask question on the displayed table
                # but2 = st.button("Ask Question on displayed Table", key='but2')

                
                # if but2:
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
        except NameError:
                st.info("Select Table to Display")



        # #else if user wants a range of tables
        # elif value == 'Range of Tables':
        #     #get the lower range
        #     lower_range = st.sidebar.selectbox("Select lower Range", options=pages)
        #     #only when lower_range is given, should the user be asked to provide upper range
        #     if lower_range is not None:
        #         #user selects upper range
        #         upper_range = st.sidebar.selectbox("Select upper Range", options=pages)
        #     but2 = st.sidebar.button("Click to view Tables")
        #     if but2:
        #         #display the range of tables    
        #         tables = multi_table_disp(doc=doc, pages=pages, lower_range=int(lower_range), upper_range=int(upper_range))
        #     st.warning("Generally, it is best practice to pass a single Table to the LLM")










    