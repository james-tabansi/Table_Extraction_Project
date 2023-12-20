import fitz
import pprint
import streamlit as st
import numpy as np
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
#to get the word-embeddings for the tokens
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

#to create vectors from the embeddings
from langchain.vectorstores import FAISS

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import TapasTokenizer, TapasForQuestionAnswering


#use the langchain framework to build apps powered by LLM
#to get the word-embeddings for the tokens using openai embeddings
from langchain.embeddings.openai import OpenAIEmbeddings

#to create vectors from the embeddings
from langchain.vectorstores import FAISS

from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TapasConfig, TapasForQuestionAnswering, TapasTokenizer, AutoModelForTableQuestionAnswering




def pdf_reader(path):

    #read the pdf
    pdf = fitz.open(path)

    if pdf is not None:
        st.success('PDF File Uploaded Successfully')

    return pdf

def get_table_objects(pdf):
    """
    read the pdf document and obtain all the table objects inside of it
    returns list of table object
    args:
        pdf: uploaded pdf file
    
    """
    #list to store table objects
    table_object = []

    #get the index 
    ind = []

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
        #for each pages in the document
        for page in doc:
            #check if there is a table
            tab = page.find_tables()
            #if there are at least one table object in the page
            if len(tab.tables) > 0:
                #for each of the table objects
                for i in tab.tables:
                    #append it to the list
                    table_object.append(i)
        
        for i in range(1, len(table_object)+1):
            ind.append(i)
                    

    return table_object, ind

def convert_table_to_df(table_object, selection):
    """
    convers the table to a dataframe
    """
    #create a string variable to save the string content of the table
    content = ""
    #convert the table to string
    df = table_object[selection].to_pandas()
    # #df = l1[page_num-1].to_pandas()

    return df

def convert_table_to_string(table_object, selection):
    """
    convers the table to a string
    """
    #create a string variable to save the string content of the table
    content = ""
    #convert the table to string
    df = table_object[selection].to_pandas()
    string = df.to_string()
    content += string
    return content


def TapasQAprompter(query, content):
    """
    generates prompt with user's query
    args:
        query: user's question
        content: content of table in a string format  
    """

    #make the prompt
    prompt1 = f"You are a good assistant that receives content of a table as a string and answer user's questions from the content by performing numeric computation using the content.\n"
    prompt2 = f"Example: content: Term Undergraduate Graduate 0 Fall 2019 19886 3441 1 Winter 2020 19660 3499 2 Spring 2020 19593 3520, question: what is the sum of the values in the second column; answer:19886+19660+19593=59139"
    prompt3 = f"Here is the question: {query}"

    prompt = prompt1+prompt2+prompt3

    tokenizer = TapasTokenizer.from_pretrained("Tapas4QA/Tapas4QATokinzer")
    model = TapasForQuestionAnswering.from_pretrained("Tapas4QA/Tapas4QA2")

    inputs = tokenizer(table=content, queries=[prompt], padding="max_length", return_tensors="pt")
    outputs = model(**inputs)
    predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
    inputs, outputs.logits.detach(), outputs.logits_aggregation.detach(), cell_classification_threshold=0.7)

    #return outputs

    #st.write(outputs)

    # let's print out the results:

    answers = []
    for coordinates in predicted_answer_coordinates:
        if len(coordinates) == 1:
            # only a single cell:
            answers.append(content.iat[coordinates[0]])
        else:
            # multiple cells
            cell_values = []
            for coordinate in coordinates:
                cell_values.append(content.iat[coordinate])
            answers.append(", ".join(cell_values))
    


    print("")
    for query, answer in zip(query, answers):
        print(query)
        print("Predicted answer: " + answer)

    return answers




def prompter(query, content):
    """
    generates prompt with user's query
    args:
        query: user's question
        content: content of table in a string format  
    """

    #make the prompt
    prompt1 = f"You are a good assistant that receives content of a table as a string and answer user's questions from the content by performing numeric computation using the content.\n"
    prompt2 = f"Example: content: Term Undergraduate Graduate 0 Fall 2019 19886 3441 1 Winter 2020 19660 3499 2 Spring 2020 19593 3520, question: what is the sum of the values in the second column; answer:19886+19660+19593=59139"
    prompt3 = f"Here is the question: {query}, use the content of this table {content}"

    prompt = prompt1+prompt2+prompt3

    inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")
    outputs = model(**inputs)
    predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
    inputs, outputs.logits.detach(), outputs.logits_aggregation.detach()
)


    #--- LArge FLAN-T5-----
    # model_dir = 'Large_Flan_T5/large_FLAN_T5'
    # tokenizer_dir = 'Large_Flan_T5/large_FLAN_T5_Tokenizer'

    #--- FLAN-T5 ----
    # model_dir = 'Flan-T5/FLAN_T5'
    # tokenizer_dir = 'Flan-T5/FLAN_T5_Tokenizer'

    #-- Tapas ----e
    # model_dir = 'TAPAS/Tapas_model'
    # tokenizer_dir = 'TAPAS/Tapas_Tokenizer'

    # config = TapasConfig("google-base-finetuned-wikisql-supervised")
    # tokenizer = TapasTokenizer.from_pretrained("google/tapas-base", config=config)
    # model = TapasForQuestionAnswering("google/tapas-base", config=config)

    #-- AutoTokenizer---
    # tokenizer = AutoTokenizer.from_pretrained("AutoTokenizer")
    # model = AutoModelForSeq2SeqLM.from_pretrained("AutoModelS2S")


    # Load the tokenizer and model from the specified directory
    # tokenizer = T5Tokenizer.from_pretrained(tokenizer_dir)
    # model = T5ForConditionalGeneration.from_pretrained(model_dir)

    # tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    # model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

    input_ids = tokenizer.encode(table =content , query=prompt, return_tensors="pt")
    st.write(input_ids)
    outputs = model.generate(**input_ids)
    st.write(outputs)
    answer = tokenizer.decode(outputs, skip_special_tokens=True)
    #answer = tokenizer.decode(tokenizer.get_answer(outputs["logits"], logits=True))

    st.write(answer)



    # #get the pyTorch tensor
    # input_ids = tokenizer.encode(prompt, return_tensors="pt")
                
    # #get the output
    # outputs = model.generate(input_ids)
    # #get the answer decoded to human readable language
    # answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer
