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
st.write("## Welcome to Table Extraction Web Application")


#allow pdf file upload
pdf = st.file_uploader("Upload your pdf", type='pdf')

from functions import get_table_objects, convert_table_to_string, prompter, TapasQAprompter
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



#display user's selected table



# from utility_fns import (pdf_reader, page_finder, table_finder, single_table_disp, multi_table_disp, 
# table_2_llm, input_qa)

# #list to store table objects
# l1 = []

# #if pdf is uploaded
# if pdf is not None:
#     #get the name of the pdf file
#     name = pdf.name

#     #create a temporal file path to locate the pdf
#     temp_file_path = os.path.join(tempfile.mkdtemp(), str(name))

#     #save the pdf file inside a temporal folder such that the file path will now exist
#     with open(temp_file_path, "wb") as temp_file:
#         temp_file.write(pdf.read())
    

#     doc = pdf_reader(temp_file_path)
#     #for each pages in the document
#     for page in doc:
#         #check if there is a table
#         tab = page.find_tables()
#         #if there are at least one table object in the page
#         if len(tab.tables) > 0:
#             #for each of the table objects
#             for i in tab.tables:
#                 #append it to the list
#                 l1.append(i)
                


#     #provided the document has been read and there is at least 1 table in the PDF
#     if doc is not None:
#         ind = []
#         for i in range(1, len(l1)+1):
#             ind.append(i)


#         #get the page number
#         page_num = st.sidebar.selectbox("Select page to view Table content", options=ind)
#         but1 = st.sidebar.button("Click to view Table")
#         try:
#             if but1:
#                 table_data = l1[page_num-1].extract()
#                 #display it as table on streamlit
#                 st.table(table_data)

#             #ask user if they want to query the table
#             option = st.button("Ask A Question On This Table")
                
#             #preprocess the table
#             if option:
#                 #create a string variable to save the string content of the table
#                 content = ""
#                 #convert the table to string
#                 df = l1[page_num-1].to_pandas()
#                 string = df.to_string()
#                 content += string
#                 st.write(content)
                


#                 #get user question
#                 query = st.sidebar.text_input("Ask questions about the Tables in the PDF")


#                 #append the context and query as a text input
#                 text = input_qa(context="This is the content of a table {content}", question=query)

#                 #get the pyTorch tensor
#                 input_ids = tokenizer.encode(text, return_tensors="pt")
                
#                 #get the output
#                 outputs = model.generate(input_ids)

#                 answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

#                 #display the output
#                 st.write(tokenizer.decode(outputs[0], skip_special_tokens=True))
                
#         except NameError:
#                 pass































                # #split the content into tokens so it is not too large for the model to process
                # #'\n' uses the new line character as the separator
                # text_splitter = CharacterTextSplitter(separator="\n",chunk_size=1000, 
                #             chunk_overlap=200,
                #             length_function=len)
                # st.write(content)


                #pass the content of the table to the splitter to obtain tokens
                #tokens = text_splitter.split_text(content)

                # #ask the user if he wants to ask question on the displayed table
                # but2 = st.button("Ask Question on displayed Table", key='but2')

                
                # if but2:
                #get the user to give prompts or questions on the pdf


                # if query is not None:
                #     #if user inputs a query
                #     #take the query and search the knowledge base for content related to user quuery
                #     information = knowledge_base.similarity_search(query)

                #     #pass the found information to the openai model
                #     #initialize the model
                #     model = OpenAI() #here you can specify the type of model

                #     #load the question answering chain using the openai model
                #     chain = load_qa_chain(llm=model, chain_type="stuff")

                #     #run the chain with the information and the user query to generate a response
                #     response = chain.run(input_documents=information, question=query)

                #     #display the response generated by the openai model as a success
                #     st.success(response)

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










    