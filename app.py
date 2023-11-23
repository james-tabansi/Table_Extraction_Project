## Import relevant packages

#for building user interface
import streamlit as st

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

