import fitz
import pprint
import streamlit as st
import numpy as np
#use the langchain framework to build apps powered by LLM
#to split the texts inside the pdf into smaller chunks
from langchain.text_splitter import CharacterTextSplitter

#load the model
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI



#use the langchain framework to build apps powered by LLM
#to get the word-embeddings for the tokens using openai embeddings
from langchain.embeddings.openai import OpenAIEmbeddings

#to create vectors from the embeddings
from langchain.vectorstores import FAISS

def table_viewer(path):
    """
    reads a pdf from path, displays the tables found in the document and returns a string
    variable that contains the content of each table
    """
    #read the pdf document
    doc = fitz.open(path)
    #save a list variable to append the content of the table as a string
    l1 = []
    #iterate through the pages of the document
    for page_num in range(len(doc)):

        #for each of the pages
        page = doc[page_num]
        #locate and extract the table if found in page
        table = page.find_tables()

        #if table exists in that page, display it
        if table.tables:
            #iterate through the tables found in that page
            for i in range(len(table.tables)):
                #write the information of the table and the page it is found in
                st.write(f"Table {i + 1} in page {page_num}")
                #extract the table
                table_data = table[i].extract()
                #display it as table on streamlit
                st.table(table_data)
                #convert the table to a datframe
                df = table[i].to_pandas()
                #convert the dataframe to a string
                string = df.to_string()
                #append the string to the list
                l1.append(string)
    #convert the list to a string
    table_content = ' '.join(l1)

    return table_content

def pdf_reader(path):

    #read the pdf
    pdf = fitz.open(path)

    if pdf is not None:
        st.success('PDF File Uploaded Successfully')

    return pdf

def page_finder(doc):
    """
    returns the page number where tables are found for selection
    args:
        doc: pdf document
    """
    #save a list variable to stoe the pages where tables are found
    l1 = []
    #iterate through the pages of the document
    for page_num in range(len(doc)):

        #for each of the pages
        page = doc[page_num]
        #locate and extract the table if found in page
        table = page.find_tables()

        #if table exists in that page, append it to the list
        if table.tables:
            l1.append(page_num)
    
    return l1

def table_finder(doc):
    """
    returns the number of tables found in all the pages of the pdf
    args:
        doc: pdf document
    """
    #count the number of tables in each pages of the pdf
    count = 0
    #iterate through the pages of the document
    for page_num in range(len(doc)):

        #for each of the pages
        page = doc[page_num]
        #locate and extract the table if found in page
        table = page.find_tables()

        #if table exists in that page, display it
        if table.tables:
            #iterate through the tables found in that page
            for i in range(len(table.tables)):
                count = count + 1
    return count

def single_table_disp(doc, page_num):
    """
    displays a single table based on user's selection
    args:
        doc: pdf document
        page_num : (int) page number
    
    """
    page = doc[page_num]

    table = page.find_tables()

    for i in range(len(table.tables)):
        #extract the table
        table_data = table[i].extract()
        #display it as table on streamlit
        st.table(table_data)
    return table

def multi_table_disp(doc,pages, lower_range, upper_range):
    """
    displays multiple table based on user's selection
    args:
        doc: pdf document
        pages: iterable. list of page numbers
        lower_range: (int) lower page index
        upper_range: (int) upper page index
    
    I am tackling this problem by first creating an empty list to save the index
    I will get the index of the starting position and that of the last
    I will append the values of for each of those positions to the empty list created
    Now I can loop over that new list and get the exact range of page number containing the tables
    """
    #empty list to store values
    l2 = []

    #get the index position for the lower value
    lower_index = pages.index(lower_range)
    #get the index position for the upper value
    upper_index = pages.index(upper_range)

    #loop through this indeces
    for i in range(lower_index, upper_index + 1):
        #append the value of the page number to the new list
        l2.append(pages[i])

    #for each page_number in the new list
    for page_num in l2:
        #display the table's page number
        st.write(f"Table in Page {page_num}")
        #get the page from the document
        page = doc[page_num]

        #find the tables in that page
        table = page.find_tables()

        #for each tables in the page
        for i in range(len(table.tables)):
            #extract the table
            table_data = table[i].extract()
            #display it as table on streamlit
            st.table(table_data)
    return table

def table_2_llm(table):
    #create a string variable to save the string content of the table
    content = ""
    #convert the table to string
    for i in range(len(table.tables)):
        #if there is just one table on the page selected
        if len(table.tables) == 1:
            #extract the table
            table_data = table[i].extract()
            #convert the table to a datframe
            df = table[i].to_pandas()
            #convert the dataframe to a string
            string = df.to_string()
            #join the content to the variable
            content += string
        #but if there are more than one tables on the page
        elif len(table.tables) > 1:
            string1 = f"Start of Table {i + 1} "
            #extract the table
            table_data = table[i].extract()
            #convert the table to a datframe
            df = table[i].to_pandas()
            #convert the dataframe to a string
            string2 = df.to_string()
            string3 = f" end of Table {i + 1} /n"
            #join the content to the variable
            content += (string1 + string2 + string3)

    return content

def input_qa(question, context):
    text = f"Here is the context: {context}, use it to answer the question {question}"
    return text
