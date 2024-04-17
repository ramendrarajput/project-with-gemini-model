import PyPDF2
import streamlit as st
from dotenv import load_dotenv
##from google.cloud import texttospeech
##from pydub import AudioSegment
##from pydub.playback import play
from google.generativeai import GenerativeModel
import google.generativeai as genai
import os 
def get_gemini_response_t(question):
    model = GenerativeModel('gemini-pro')
    response = model.generate_content(question)
    return response.text

def pdf_proc():
 # Open the PDF file
 #uploaded_file = st.file_uploader("Choose a pdf file...", type=["pdf"])
 
 with open("C:\Users\ramen\Downloads\0394 अ 6 कुंजल पत्नी शैलेंद्र जैन बैनामा खैरीकला.pdf", 'rb') as file:
    reader = PyPDF2.PdfFileReader(file)

    # Get the number of pages in the PDF file
    num_pages = reader.getNumPages()

    # Loop through the pages and extract the text
    for i in range(num_pages):
        page = reader.getPage(i)
        text = page.extractText()
        print(text)

    # Create a variable to store the extracted text
    extracted_text = ""

    # Loop through the pages and extract the text
    for i in range(num_pages):
        page = reader.getPage(i)
        text = page.extractText()
        extracted_text += text
    st.write(extracted_text)
    # Print the extracted text
    #print(extracted_text)

def main():
   try:
        load_dotenv()  # take environment variables from .env
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
        ##initialize our streamlit app
        st.set_page_config(page_title="Q&A Demo.")
        st.subheader("Advanced Chat Application")
        st.caption("Developer: Ramendra singh rajput")
        pdf_proc()

   except IOError as e:
        print(f"An error occurred: {e}")    

if __name__ == "__main__":
   main()
