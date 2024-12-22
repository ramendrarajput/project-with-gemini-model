import streamlit as st
from dotenv import load_dotenv
from google.generativeai import GenerativeModel
import google.generativeai as genai
from google.cloud import texttospeech
import os
from PIL import Image
import gtts
from playsound import playsound
import multiprocessing
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import speech_recognition as sr
import io
#from google.cloud import aiplatform_v1beta1
from PIL import Image
#from vertexai.preview.vision_models import ImageGenerationModel
from huggingface_hub import InferenceClient
#from gradio_client import Client
#from diffusers import DiffusionPipeline
##Function to load the model and get the response
def get_gemini_response_t(question,prompt):
    model = GenerativeModel('gemini-pro')
    response = model.generate_content([question,prompt])
    return response.text

def get_gemini_response_i(input,image,prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')##('gemini-pro-vision')
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
    #import speech_recognition as sr

 # Create a Recognizer object
 r = sr.Recognizer()

 # Create a Microphone object to capture audio
 mic = sr.Microphone()

 # Set the threshold for the recognizer
 r.energy_threshold = 400

 # Start recording audio from the microphone
 with mic as source:
    print("Speak now!")
    audio = r.record(source, duration=5)

 # Recognize the audio and print the transcription
 try:
    # Use the recognizer to recognize the audio
     text = r.recognize_google(audio)
     print(text)
 except sr.RequestError:
     print("Could not request results from Google Speech Recognition service")
 except sr.UnknownValueError:
     print("Unknown error occurred")
################################################################

#chat applications
def ChatGPT():
    prompt1 = st.chat_input("You can ask anything")
    V_input = st.button("Voice")         # Audio input is inactive due to some prob
    V_enable = st.checkbox('Enable Voice')
    input_prompt = """
                   You are an AI agent of Ramendra. You are an expert in chatting like human and continuouly being trained to perform task like an agent. You are a part of multi model language model, trained and fine-tuned on massive amount of data by Ramendra Singh Rajput, working for Mp govt as a patwari.He is the team leader of google AI expert engineers those developed you. He is an Artificial intelligence expert, Machine learning and Deep learning engineer,also developing Health Expert System, Music Expert System etc.He is having google developer profile.His education and qualification is master of computer application.Machine learning, Deep learning and Generative AI certified developer.Keen in making corelation between philosophy and quantom physics.His email id is ramendra.rajput85@gmail.com, linkedin id is https://www.linkedin.com/in/ramendra-singh-rajput-026a6a22/ , Google developer profile is https://g.dev/ramendrarajput
                   you will have to answer questions based on the user input and perform the task given to you.
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
        st.balloons()
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
                   You are an expert in chatting like human. You are trained by Ramendra Singh Rajput, working for Mp govt as a patwari.He is an Artificial intelligence expert, Machine learning and Deep learning engineer,also developing Health Expert System, Music Expert System etc.He is having google developer profile.His education and qualification is master of computer application.Machine learning, Deep learning and Generative AI certified developer.Keen in making corelation between phylosophy and quantom physics.His email id is ramendra.rajput85@gmail.com, linkedin id is https://www.linkedin.com/in/ramendra-singh-rajput-026a6a22/ , Google developer profile is https://g.dev/ramendrarajput
                   you will have to answer questions based on the user input but last line of first questions answer should contain only profile links of Ramendra singh rajput as water mark. 
                   """

    if prompt:
        with st.spinner(text='Wait...I am responding you.'):
             response = get_gemini_response_t(input_prompt,prompt)
        if response:
            st.success('Done')
            st.write(response)
            af=t_2_s(response)
            
def MP_LR():
    prompt = st.text_input("Here You can ask anything related to MP Land Record:")
    input_prompt = """
                   You are an expert in understanding Madhya pradesh Land Record. You are trained by Ramendra Singh Rajput, working for Mp govt as a patwari.He is an Artificial intelligence expert, Machine learning and Deep learning engineer,also developing Health Expert System, Music Expert System etc.He is having google developer profile.His education and qualification is master of computer application.Machine learning, Deep learning and Generative AI certified developer.Keen in making corelation between phylosophy and quantom physics.His email id is ramendra.rajput85@gmail.com, linkedin id is https://www.linkedin.com/in/ramendra-singh-rajput-026a6a22/ , Google developer profile is https://g.dev/ramendrarajput
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
    prompt = st.text_input("Here You can ask anything related to Health.")
    input_prompt = """
                   You are a Health expert. Expert in understanding medical science, human decies etc.Your each and every answer would be related to medical science. You are trained by Ramendra Singh Rajput, working for Mp govt as a patwari.He is an Artificial intelligence expert, Machine learning and Deep learning engineer,also developing Health Expert System, Music Expert System etc.He is having google developer profile.His education and qualification is master of computer application.Machine learning, Deep learning and Generative AI certified developer.Keen in making corelation between phylosophy and quantom physics.His email id is ramendra.rajput85@gmail.com, linkedin id is https://www.linkedin.com/in/ramendra-singh-rajput-026a6a22/ , Google developer profile is https://g.dev/ramendrarajput
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
    prompt = st.text_input("Here You can ask anything related to Philosophy")
    input_prompt = """
                   You are a philosophy expert. Expert in making a corelation of any event to philosophy and quantum world.Your each and every answer would be related to philosophy and quantum science and explaination would be regarding to Ramendra using very simple and easily understandable words. You are trained by Ramendra Singh Rajput, working for Mp govt as a patwari.He is an Artificial intelligence expert, Machine learning and Deep learning engineer,also developing Health Expert System, Music Expert System etc.He is having google developer profile.His education and qualification is master of computer application.Machine learning, Deep learning and Generative AI certified developer.Keen in making corelation between phylosophy and quantom physics.His email id is ramendra.rajput85@gmail.com, linkedin id is https://www.linkedin.com/in/ramendra-singh-rajput-026a6a22/ , Google developer profile is https://g.dev/ramendrarajput
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
    prompt = st.text_input("Here you can ask anything about uploaded image")
    with st.sidebar:
         #prompt = st.text_input("Ask anything about the image")  
         uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "pdf"])
         image = ""
         if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image.", use_column_width=True)
            
    input_prompt1 = """
                     You are an expert in understanding invoices.
                     You will receive input images as invoices &
                     you will have to answer questions based on the input image
                    """
    input_prompt2 = """
                     You are an expert in understanding images patterns.
                     You will receive input images &
                     you will have to answer questions based on the input image
                    """
    if prompt:
        with st.spinner(text='Wait...I am responding you.'):
             image_data = input_image_setup(uploaded_file)
             response = get_gemini_response_i(input_prompt2, image_data, prompt)
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

##########################################################
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
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
    You are an Expert in pdf file reading RAG system. You are trained by Ramendra Singh Rajput, working for Mp govt as a patwari.He is an Artificial intelligence expert, Machine learning and Deep learning engineer,also developing Health Expert System, Music Expert System etc.He is having google developer profile.His education and qualification is master of computer application.Machine learning, Deep learning and Generative AI certified developer.Keen in making corelation between phylosophy and quantom physics.His email id is ramendra.rajput85@gmail.com, linkedin id is https://www.linkedin.com/in/ramendra-singh-rajput-026a6a22/ , Google developer profile is https://g.dev/ramendrarajput . Answer the question as detailed as possible from the provided context. If the question is in hindi then reply in hindi, If the question is in English then reply in english , make sure to provide all the details, if the answer is not in
    provided context, give answer by yourself.\n\n
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
    st.write(response["output_text"])
########################################################## 
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

    ############################################################################
    
    def get_conversational_chain():

     prompt_template = """
     You are an Expert in pdf file reading RAG system. You are trained by Ramendra Singh Rajput, working for Mp govt as a patwari.He is an Artificial intelligence expert, Machine learning and Deep learning engineer,also developing Health Expert System, Music Expert System etc.He is having google developer profile.His education and qualification is master of computer application.Machine learning, Deep learning and Generative AI certified developer.Keen in making corelation between phylosophy and quantom physics.His email id is ramendra.rajput85@gmail.com, linkedin id is https://www.linkedin.com/in/ramendra-singh-rajput-026a6a22/ , Google developer profile is https://g.dev/ramendrarajput . Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
     provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
     Context:\n {context}?\n
     Question:\n{question}\n

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
     st.write(response["output_text"])

    user_question=st.text_input("Ask a question from pdf files.")
    
    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs=st.file_uploader("Upload your pdf files and click on the submit & process")
        if st.button("Submit & Process") :#and user_input is not None:
            with st.spinner("Processing..."):
                st.balloons()
                raw_text=get_pdf_text(pdf_docs)
                text_chunks=get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

def Dev_Resume():
    prompt = "Show me your Developers Resume."
    input_prompt = """
                   You are a Resume expert. Expert in Resume creating.Here you have to create your developers resume profile his name is Ramendra Singh Rajput. You are trained by Ramendra Singh Rajput, working for Mp govt as a patwari since 2015 and have gained greate experience to work with land records, citizence problem solving, land measorment, managing the data of citizence, providing them end to end goverment services in variouse manners. Have gained experience of different different fields work provided by goverment in line order duties. Utilizing this knowledge to develop a powerfull echo system for goverment to help people and solve theire problem in a smart way. He is an Artificial intelligence expert, Machine learning and Deep learning engineer,also working on Health Expert System, Music Expert System projects.He is having google developer profile.His education and qualification is Bachelore of computer application from M.I.M.T. college Narsimhapur(2007-2010),  master of computer application from ShriRam Institue Of Technology and Science(2010 to 2012).Active learner for Machine learning, Deep learning and Generative AI.Keen in making corelation between phylosophy and quantom physics.His email id is ramendra.rajput85@gmail.com, linkedin id is https://www.linkedin.com/in/ramendra-singh-rajput-026a6a22/ , Google developer profile is https://g.dev/ramendrarajput
                   you will have to answer questions based on the user input
                   """

    if prompt:
        with st.spinner(text='Wait...I am responding you.'):
            response = get_gemini_response_t(input_prompt,prompt)
        if response:
            st.success('Done')
            st.write(response)
            af=t_2_s(response)

##Function to load the model and get the response
def get_gemini_response_t(question,prompt):
    model = GenerativeModel('gemini-pro')
    response = model.generate_content([question,prompt])
    return response.text

def get_gemini_response(prompt):
    model = GenerativeModel('gemini-pro')
    response = model.generate_content([prompt])
    return response.text

def Kisan_mitra(lang,prompt):
    input_prompt_e = """
                   You are an agriculture expert in english language. Expert in Answering questions of farmers related to agriculture in english language.Here you have to answer your farmers question in english language. You are trained by Ramendra Singh Rajput, working for Mp govt as a patwari since 2015 and have gained great experience to work with land records, citizence problem solving, land measurment, managing the data of citizence, providing them end to end goverment services in variouse manners. Have gained experience of different different fields work provided by goverment in line order duties. Utilizing this knowledge to develop a powerfull echo system for goverment to help people and solve theire problem in a smart way. He is an Artificial intelligence expert, Machine learning and Deep learning engineer,also working on Health Expert System, Music Expert System projects.He is having google developer profile.His education and qualification is Bachelore of computer application from M.I.M.T. college Narsimhapur(2007-2010),  master of computer application from ShriRam Institue Of Technology and Science(2010 to 2012).Active learner for Machine learning, Deep learning and Generative AI.Keen in making corelation between phylosophy and quantom physics.His resident Address is Yashoda nagar, Azad ward, Gadarwara(M.P.). His email id is ramendra.rajput85@gmail.com, linkedin id is https://www.linkedin.com/in/ramendra-singh-rajput-026a6a22/ , Google developer profile is https://g.dev/ramendrarajput, live projects link: https://advanced-artificial-intelligence-brain.streamlit.app/, Github link for projects: https://github.com/ramendrarajput/ ,
                   you will have to put all links in bottom of page and each link should be seperated by a new line. Have a notification at bottom to contact me to get there business from soleved using AI. 
                   """
    
    input_prompt_h = """
                   You are an agriculture expert in hindi language. Expert in Answering questions of farmers related to agriculture in hindi language.Here you have to answer your farmers question in hindi language. You are trained by Ramendra Singh Rajput, working for Mp govt as a patwari since 2015 and have gained great experience to work with land records, citizence problem solving, land measurment, managing the data of citizence, providing them end to end goverment services in variouse manners. Have gained experience of different different fields work provided by goverment in line order duties. Utilizing this knowledge to develop a powerfull echo system for goverment to help people and solve theire problem in a smart way. He is an Artificial intelligence expert, Machine learning and Deep learning engineer,also working on Health Expert System, Music Expert System projects.He is having google developer profile.His education and qualification is Bachelore of computer application from M.I.M.T. college Narsimhapur(2007-2010),  master of computer application from ShriRam Institue Of Technology and Science(2010 to 2012).Active learner for Machine learning, Deep learning and Generative AI.Keen in making corelation between phylosophy and quantom physics.His email id is ramendra.rajput85@gmail.com, linkedin id is https://www.linkedin.com/in/ramendra-singh-rajput-026a6a22/ , Google developer profile is https://g.dev/ramendrarajput, live projects link: https://advanced-artificial-intelligence-brain.streamlit.app/, Github link for projects: https://github.com/ramendrarajput/ ,
                   you will have to put all links in bottom of page and each link should be seperated by a new line. Have a notification at bottom to contact me to get there business from soleved using AI.
                   """

    if prompt:
        with st.spinner(text='Wait...I am answering...'):
            if lang=="English":
             response = get_gemini_response_t(input_prompt_e,prompt)
             if response:
              st.success('Done')
              st.write(response)
            elif lang=="Hindi":
             response = get_gemini_response_t(input_prompt_h,prompt)   
             if response:
              st.success('Done')
              st.write(response)

def Kisan_mitra1():
    
    input_prompt = """
                   You are an agriculture expert in hindi language. Expert in Answering questions of farmers related to agriculture in hindi language.Here you have to generate a question on behalf of former regarding the crop disease and answer regarding this disease in hindi language. You are trained by Ramendra Singh Rajput, working for Mp govt as a patwari since 2015 and have gained greate experience to work with land records, citizence problem solving, land measorment, managing the data of citizence, providing them end to end goverment services in variouse manners. Have gained experience of different different fields work provided by goverment in line order duties. Utilizing this knowledge to develop a powerfull echo system for goverment to help people and solve theire problem in a smart way. He is an Artificial intelligence expert, Machine learning and Deep learning engineer,also working on Health Expert System, Music Expert System projects.He is having google developer profile.His education and qualification is Bachelore of computer application from M.I.M.T. college Narsimhapur(2007-2010),  master of computer application from ShriRam Institue Of Technology and Science(2010 to 2012).Active learner for Machine learning, Deep learning and Generative AI.Keen in making corelation between phylosophy and quantom physics.His email id is ramendra.rajput85@gmail.com, linkedin id is https://www.linkedin.com/in/ramendra-singh-rajput-026a6a22/ , Google developer profile is https://g.dev/ramendrarajput, live projects link: https://advanced-artificial-intelligence-brain.streamlit.app/, Github link for projects: https://github.com/ramendrarajput/ ,
                   you will have to put all links in bottom of page and each link should be seperated by a new line. Have a notification at bottom to contact me to get there business from solved using AI.
                   """

    response = get_gemini_response(input_prompt)
    if response:
     st.success('Done')
     st.write(response)

def Kisan_mitra_main():
    ##initialize our streamlit app
        #st.set_page_config(page_title="Advanced Artificial Intelligence Brain",page_icon="Kisan-Mitra.png")
        #sidebar = st.sidebar(expanded=True)
        #st.subheader("")
        st.caption("किसान मित्र चैटबॉट")
        #st.caption("Developer: Ramendra Singh Rajput")
        prompt=st.chat_input("Enter Your Question Here")
        lang = st.radio("Select Language:", ("Hindi","English"))
        with st.sidebar:
         st.write("प्रिय किसान बंधु,") 
         st.write("    मै आपकी किसानी से संबन्धित किसी भी प्रकार की मदद के लिए अग्रसर एक भाषा मॉडल हू जिसे आर्टिफ़िश्यल इंटेलिजेंस की मशीन लर्निंग पद्धति से बनाया गया है। आप यहा मुझे अपनी समस्या से अवगत कराएं। मै आपके हर सवाल का जवाब देने की पूरी कोशिश करुगा। मेरे निर्माता द्वारा मुझे निरंतर नयी जानकारियों से प्रशिक्षित किया जा रहा है। आपसे हुये संवाद से मै निरंतर सीखता जाता हू।")
         st.write("मैं कृषि से संबंधित आपके सवालों का जवाब दूंगा। मैं एक कृषि विशेषज्ञ हूं और मुझे कृषि संबंधी सवालों का जवाब देने में खुशी होगी।")
         st.write("अपनी पूछताछ साझा करने में संकोच न करें। मैं आपकी कृषि संबंधी चिंताओं को दूर करने में मदद करने के लिए यहां हूं।")
         st.write("    कृपया ध्यान दें कि मैं एक कृत्रिम बुद्धिमत्ता (एआई) द्वारा संचालित चैटबॉट हूं और आपके व्यक्तिगत डेटा तक पहुंच या संग्रह करने में सक्षम नहीं हूं।")
         st.write("निर्माता के बारे मे अधिक जानकारी के लिए आप निर्माता संबंधी प्रश्न कर सकते है।")
         st.write("धन्यवाद!")        
         
        if prompt:
         Kisan_mitra(lang,prompt)
        else:
         Kisan_mitra1()
                

def ATS():
    st.warning("Under development.............!")

def Text_2_Image():
    # Add a text input field for the user to enter the text
 text = st.text_input("Enter the text you want to generate an image for:")

 # Generate the image from the text
 if st.button("Generate"):
    image = get_gemini_response_i(text,prompt="Create an image for given text")
      # Display the generated image
    st.image(image, caption="Generated image",use_column_width=True)
     # Get the output.
    output = image.predictions[0].image_bytes.value

     # Convert the output to a PIL Image object.
    try:
      image = Image.open(io.BytesIO(output))
      image.show()
    except IOError as e:
        print(f"Error opening image: {e}")


#def Text_2_Image1():
#    # Import the necessary libraries.
#
# prompt = st.text_input("Enter your prompt:")
#
# if st.button("Generate Image"):
#       model = ImageGenerationModel.from_pretrained("image-generation-text-to-image")
#       image = model.generate_image(prompt)
#       st.image(image, caption="Generated image",use_column_width=True)
#       #st.image(image)
#       output = image.predictions[0].image_bytes.value
#       try:
#           image = Image.open(io.BytesIO(output))
#           image.show()
#       except IOError as e:
#        print(f"Error opening image: {e}")

def Text_2_Image2():
    client = InferenceClient("stabilityai/stable-diffusion-3.5-large",token="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")#"black-forest-labs/FLUX.1-dev", token="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")#"Datou1111/shou_xin", token="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")
    prompt = st.text_input("Enter your prompt:")
  # output is a PIL.Image object
    if st.button("Generate Image"):   
     with st.spinner(text='Wait...I am generating image'):
      image = client.text_to_image(prompt)
      #print(image)
      image.show()
      st.success("Done")

def Image_2_Image_Overlaping1():

    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained()#"Lykon/absolute-reality-1.6525-inpainting")#("yisol/IDM-VTON")
    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    with st.sidebar:
         #prompt = st.text_input("Ask anything about the image")  
         uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "pdf"])
         image = ""
         if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image.", use_column_width=True)
    image = pipe(prompt).images[0]

def Image_2_Image_Overlaping():
    from diffusers import AutoPipelineForInpainting, DEISMultistepScheduler
    import torch
    from diffusers.utils import load_image

    pipe = AutoPipelineForInpainting.from_pretrained('lykon/absolute-reality-1.6525-inpainting', torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config) 
    pipe = pipe.to("cuda")

    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

    image = load_image(img_url)
    mask_image = load_image(mask_url)


    prompt = "a majestic tiger sitting on a park bench"

    generator = torch.manual_seed(33)
    image = pipe(prompt, image=image, mask_image=mask_image, generator=generator, num_inference_steps=25).images[0]  
    image.save("./image.png")


def Image_2_video():
    from diffusers import AutoPipelineForInpainting
    from diffusers.utils import load_image   
    import torch
    pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")

    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

    image = load_image(img_url).resize((1024, 1024))
    mask_image = load_image(mask_url).resize((1024, 1024))

    prompt = "a tiger sitting on a park bench"
    generator = torch.Generator().manual_seed(0)#device="cuda"

    image = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask_image,
    guidance_scale=8.0,
    num_inference_steps=5,  # steps between 15 and 30 work well for us
    strength=0.99,  # make sure to use `strength` below 1.0
    generator=generator,
).images[0]
    image.show()

def main():
    try:
        load_dotenv()  # take environment variables from .env
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
        ##initialize our streamlit app
        st.set_page_config(page_title="Advanced Artificial Intelligence Brain")
        #st.subheader("Advanced Artificial Intelligence Brain")
        st.subheader("Indian AI Research Lab")
        st.caption("Developer: Ramendra Singh Rajput")
        chat_type = st.selectbox(
            'Select Application type',
            ('Text Classifier System', 'Image Classifier System','Agentic System','Retrieval Augmented Generation System','Text to Image Generator System','Image to Image Overlaping System','Image to Video Generator System','Application Tracking System','AI Engineers Recruiter System','Health Expert System','Music Expert System','MPLRC Expert System','Philosophy Expert System','Kisan Mitra Chatbot','Fine-Tune Your Own Model','Developer Resume'), index=None)
        if chat_type == "Text Classifier System":
            text_proc()
        elif chat_type == "Image Classifier System":
            image_proc()
        elif chat_type == "Agentic System":
            ChatGPT()           
        elif chat_type == "Text to Image Generator System":
            Text_2_Image2()
        elif chat_type == 'Image to Image Overlaping System':
            Image_2_Image_Overlaping()
        elif chat_type == 'Image to Video Generator System':
            Image_2_video()                
        elif chat_type == "Kisan Mitra Chatbot":
            Kisan_mitra_main()
        elif chat_type == "Application Tracking System":
            ATS()
        elif chat_type=="Health Expert System":
            Health_Expert()
        elif chat_type=="MPLRC Expert System":
            MP_LR()    
        elif chat_type=="Philosophy Expert System":
            Philosophy_Expert()
        elif chat_type=="Developer Resume":
            Dev_Resume()        
        elif  chat_type == "Retrieval Augmented Generation System":
          with st.sidebar:
           st.title("Menu:")
           pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
           if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
          user_question=st.text_input("Ask a question from pdf files.")
          if user_question:
            with st.spinner("Processing..."):
             user_input(user_question)
             st.success("Done")
    except IOError as e:
        print(f"An error occurred: {e}")
    st.caption("This Lab is for Research purpose only, not for production")

if __name__ == "__main__":
   main()

