import fitz
import pprint
import streamlit as st


def table_viewer(path):
        #read the pdf document
    doc = fitz.open(path)

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