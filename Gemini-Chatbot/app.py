import streamlit as st
from dotenv import load_dotenv
#from google.cloud import texttospeech
from pydub import AudioSegment
from pydub.playback import play
from google.generativeai import GenerativeModel
import google.generativeai as genai
from google.cloud import texttospeech
import os
from PIL import Image
import gtts
from playsound import playsound
import pygame
import multiprocessing
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import speech_recognition as sr

##Function to load the model and get the response
def get_gemini_response_t(question,prompt):
    model = GenerativeModel('gemini-pro')
    response = model.generate_content([question,prompt])
    return response.text

def get_gemini_response_i(input,image,prompt):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([input,image[0],prompt])
    return response.text

def get_gemini_response_pdf(prompt_template):
    model = ChatGoogleGenerativeAI(model='gemini-pro',temperature=0.3)
    prompt=PromptTemplate(template=prompt_template, input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

##Function to process the user input images
def input_image_setup(uploaded_file):
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

##############################################################3
# Function to convert speech to text
def speech_to_text():
    with sr.Microphone() as source:
        print("Speak now...")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        print("You said: {}".format(text))
        return text
    except:
        print("Sorry, I didn't catch that.")
        return None
################################################################

#chat applications
def ChatGPT():
    prompt1 = st.chat_input("You can ask anything")
    V_input = st.button("Voice")
    V_enable = st.checkbox('Enable Voice')
    input_prompt = """
                   You are better then chatGPT. You are an expert in chatting like human. You are trained by Ramendra Singh Rajput, working for Mp govt as a patwari.He is the team leader of google AI expert engineers those developed you. He is an Artificial intelligence expert, Machine learning and Deep learning engineer,also developing Health Expert System, Music Expert System etc.He is having google developer profile.His education and qualification is master of computer application.Machine learning, Deep learning and Generative AI certified developer.Keen in making corelation between philosophy and quantom physics.His email id is ramendra.rajput85@gmail.com, linkedin id is https://www.linkedin.com/in/ramendra-singh-rajput-026a6a22/ 
                   you will have to answer questions based on the user input. Remember neighter you are gemini.
                   """
    
   ## If enter button is clicked
    if V_enable==True and prompt1:
        with st.spinner(text='Wait...I am responding you.'):
            response = get_gemini_response_t(input_prompt,prompt1)
        if response:
            af=t_2_s(response)
            st.write(response)
            st.success('Done')
            p = multiprocessing.Process(target=playsound(af), args=(af))
            p.start()
    elif prompt1:
        with st.spinner(text='Wait...I am responding you.'):
            response = get_gemini_response_t(input_prompt,prompt1)
        if response:
            st.success('Done')
            st.write(response)
            
    if V_enable==True and V_input:
        text = speech_to_text()
        with st.spinner(text='Wait...I am responding you.'):
            response = get_gemini_response_t(input_prompt,text)
        if response:
            af=t_2_s(response)
            st.write(response)
            st.success('Done')
            p = multiprocessing.Process(target=playsound(af), args=(af))
            p.start()
    elif V_input:
        text = speech_to_text()
        with st.spinner(text='Wait...I am responding you.'):
            response = get_gemini_response_t(input_prompt,text)
        if response:
            st.success('Done')
            st.write(response)
                     
def check():
    p = multiprocessing.Process(target=playsound("response.mp3"), args=("response.mp3"))
    if st.button("Listen"):
        p.start()
    else:
        p.stop()
    if st.checkbox("Enable Voice"):
        p.stop()

def text_proc():
    prompt = st.text_input("You can ask anything")
    input_prompt = """
                   You are an expert in chatting like human. You are trained by Ramendra Singh Rajput, working for Mp govt as a patwari.He is an Artificial intelligence expert, Machine learning and Deep learning engineer,also developing Health Expert System, Music Expert System etc.He is having google developer profile.His education and qualification is master of computer application.Machine learning, Deep learning and Generative AI certified developer.Keen in making corelation between phylosophy and quantom physics.His email id is ramendra.rajput85@gmail.com, linkedin id is https://www.linkedin.com/in/ramendra-singh-rajput-026a6a22/ 
                   you will have to answer questions based on the user input
                   """

    if prompt:
        with st.spinner(text='Wait...I am responding you.'):
            response = get_gemini_response_t(input_prompt,prompt)
        if response:
            st.success('Done')
            st.write(response)
            af=t_2_s(response)
            
def MP_LR():
    prompt = st.text_input("You can ask anything")
    input_prompt = """
                   You are an expert in understanding Madhya pradesh Land Record. You are trained by Ramendra Singh Rajput, working for Mp govt as a patwari.He is an Artificial intelligence expert, Machine learning and Deep learning engineer,also developing Health Expert System, Music Expert System etc.He is having google developer profile.His education and qualification is master of computer application.Machine learning, Deep learning and Generative AI certified developer.Keen in making corelation between phylosophy and quantom physics.His email id is ramendra.rajput85@gmail.com, linkedin id is https://www.linkedin.com/in/ramendra-singh-rajput-026a6a22/ 
                   you will have to answer questions based on the user input
                   """

    if prompt:
        with st.spinner(text='Wait...I am responding you.'):
            response = get_gemini_response_t(input_prompt,prompt)
        if response:
            st.success('Done')
            st.write(response)
            af=t_2_s(response)
            
def Health_Expert():
    prompt = st.text_input("You can ask anything")
    input_prompt = """
                   You are a Health expert. Expert in understanding medical science, human decies etc.Your each and every answer would be related to medical science. You are trained by Ramendra Singh Rajput, working for Mp govt as a patwari.He is an Artificial intelligence expert, Machine learning and Deep learning engineer,also developing Health Expert System, Music Expert System etc.He is having google developer profile.His education and qualification is master of computer application.Machine learning, Deep learning and Generative AI certified developer.Keen in making corelation between phylosophy and quantom physics.His email id is ramendra.rajput85@gmail.com, linkedin id is https://www.linkedin.com/in/ramendra-singh-rajput-026a6a22/ 
                   you will have to answer questions based on the user input
                   """

    if prompt:
        with st.spinner(text='Wait...I am responding you.'):
            response = get_gemini_response_t(input_prompt,prompt)
        if response:
            st.success('Done')
            st.write(response)
            af=t_2_s(response)
            
def Philosophy_Expert():
    prompt = st.text_input("You can ask anything")
    input_prompt = """
                   You are a philosophy expert. Expert in making a corelation of any event to philosophy and quantum world.Your each and every answer would be related to philosophy and quantum science and explaination would be regarding to Ramendra using very simple and easily understandable words. You are trained by Ramendra Singh Rajput, working for Mp govt as a patwari.He is an Artificial intelligence expert, Machine learning and Deep learning engineer,also developing Health Expert System, Music Expert System etc.He is having google developer profile.His education and qualification is master of computer application.Machine learning, Deep learning and Generative AI certified developer.Keen in making corelation between phylosophy and quantom physics.His email id is ramendra.rajput85@gmail.com, linkedin id is https://www.linkedin.com/in/ramendra-singh-rajput-026a6a22/ 
                   you will have to answer questions based on the user input
                   """

    if prompt:
        with st.spinner(text='Wait...I am responding you.'):
            response = get_gemini_response_t(input_prompt,prompt)
        if response:
            st.success('Done')
            st.write(response)
            af=t_2_s(response)
             
def image_proc():
    prompt = st.text_input("Tell me about the image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "pdf"])
    image = ""
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

    input_prompt = """
                   You are an expert in understanding invoices.
                   You will receive input images as invoices &
                   you will have to answer questions based on the input image
                   """

    if prompt:
        with st.spinner(text='Wait...I am responding you.'):
            image_data = input_image_setup(uploaded_file)
            response = get_gemini_response_i(input_prompt, image_data, prompt)
        if response:
            st.success('Done')
            st.write(response)

def t_2_s(response):

 t=response

 # Select the language for the text to be spoken in
 language = 'en'

 # Create an instance of the gTTS class
 tts = gtts.gTTS(text=t, lang=language)
 
 # Save the audio file
 audio_file = 'response.mp3'
 tts.save(audio_file)
 return audio_file
 
def ChatPdf():
    
    def get_pdf_text(pdf_docs):
        text=""
        for pdf in pdf_docs:
            pdf_reader=PdfReader(pdf)
            for page in pdf_reader.pages:
                text+=page.extract_text()
        return  text
    
    def get_text_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks
    
    def get_vector_store(text_chunks):
     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
     vector_store.save_local("faiss_index")

    def get_conversational_chain():

     prompt_template = """
     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
     provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
     Context:\n {context}?\n
     Question: \n{question}\n

     Answer:
     """

     model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

     prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

     return chain
    
    def user_input(user_question):
     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
     new_db = FAISS.load_local("faiss_index", embeddings)
     docs = new_db.similarity_search(user_question)

     chain = get_conversational_chain()

    
     response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

     print(response)
     st.write("Reply: ", response["output_text"])

    user_question=st.text_input("Ask a question from pdf files.")
    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.title("Menu:")
        pdf_docs=st.file_uploader("Upload your pdf files and click on the submit & process")
        if st.button("Submit & Process") and user_input is not None:
            with st.spinner("Processing..."):
                raw_text=get_pdf_text(pdf_docs)
                st.balloons()
                text_chunks=get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

def ATS():
    st.warning("Under development...!")

def main():
    try:
        load_dotenv()  # take environment variables from .env
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
        ##initialize our streamlit app
        st.set_page_config(page_title="Q&A Demo.")
        st.subheader("Advanced Artificial Intelligence Brain")
        st.caption("Developer: RAMENDRA SINGH RAJPUT")
        
        chat_type = st.selectbox(
            'Select Application type',
            ('Text Chatbot', 'Image Chatbot','ChatGPT','Chat with pdf files','Application Tracking System','Health Expert','Madhya pradesh Land Record Expert','Philosophy Expert'), index=None)
        if chat_type == "Text Chatbot":
            text_proc()
        elif chat_type == "Image Chatbot":
            image_proc()
        elif chat_type == "ChatGPT":
            ChatGPT()           
        elif chat_type == "Chat with pdf files":
            ChatPdf()
        elif chat_type == "Application Tracking System":
            ATS()
        elif chat_type=="Health Expert":
            Health_Expert()
        elif chat_type=="Madhya pradesh Land Record Expert":
            MP_LR()    
        elif chat_type=="Philosophy Expert":
            Philosophy_Expert()    
        
    except IOError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
   main()
