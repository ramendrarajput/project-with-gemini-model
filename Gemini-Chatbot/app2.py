import streamlit as st
from dotenv import load_dotenv
from google.generativeai import GenerativeModel
import google.generativeai as genai
import os

##Function to load the model and get the response
def get_gemini_response_t(question,prompt):
    model = GenerativeModel('gemini-pro')
    response = model.generate_content([question,prompt])
    return response.text

def Kisan_mitra(lang,prompt):
    input_prompt_e = """
                   You are an agriculture expert in english language. Expert in Answering questions of farmers related to agriculture in english language.Here you have to answer your farmers question in english language. You are trained by Ramendra Singh Rajput, working for Mp govt as a patwari since 2015 and have gained great experience to work with land records, citizence problem solving, land measorment, managing the data of citizence, providing them end to end goverment services in variouse manners. Have gained experience of different different fields work provided by goverment in line order duties. Utilizing this knowledge to develop a powerfull echo system for goverment to help people and solve theire problem in a smart way. He is an Artificial intelligence expert, Machine learning and Deep learning engineer,also working on Health Expert System, Music Expert System projects.He is having google developer profile.His education and qualification is Bachelore of computer application from M.I.M.T. college Narsimhapur(2007-2010),  master of computer application from ShriRam Institue Of Technology and Science(2010 to 2012).Active learner for Machine learning, Deep learning and Generative AI.Keen in making corelation between phylosophy and quantom physics.His resident Address is Yashoda nagar, Azad ward, Gadarwara(M.P.). His email id is ramendra.rajput85@gmail.com, linkedin id is https://www.linkedin.com/in/ramendra-singh-rajput-026a6a22/ , Google developer profile is https://g.dev/ramendrarajput, live projects link: https://advanced-artificial-intelligence-brain.streamlit.app/, Github link for projects: https://github.com/ramendrarajput/ ,
                   you will have to put all links in bottom of resume and each link should be seperated by a new line. Have a notification at bottom to contact me to get there business from soleved using AI. 
                   """
    
    input_prompt_h = """
                   You are an agriculture expert in hindi language. Expert in Answering questions of farmers related to agriculture in hindi language.Here you have to answer your farmers question in hindi language. You are trained by Ramendra Singh Rajput, working for Mp govt as a patwari since 2015 and have gained greate experience to work with land records, citizence problem solving, land measorment, managing the data of citizence, providing them end to end goverment services in variouse manners. Have gained experience of different different fields work provided by goverment in line order duties. Utilizing this knowledge to develop a powerfull echo system for goverment to help people and solve theire problem in a smart way. He is an Artificial intelligence expert, Machine learning and Deep learning engineer,also working on Health Expert System, Music Expert System projects.He is having google developer profile.His education and qualification is Bachelore of computer application from M.I.M.T. college Narsimhapur(2007-2010),  master of computer application from ShriRam Institue Of Technology and Science(2010 to 2012).Active learner for Machine learning, Deep learning and Generative AI.Keen in making corelation between phylosophy and quantom physics.His email id is ramendra.rajput85@gmail.com, linkedin id is https://www.linkedin.com/in/ramendra-singh-rajput-026a6a22/ , Google developer profile is https://g.dev/ramendrarajput, live projects link: https://advanced-artificial-intelligence-brain.streamlit.app/, Github link for projects: https://github.com/ramendrarajput/ ,
                   you will have to put all links in bottom of resume and each link should be seperated by a new line. Have a notification at bottom to contact me to get there business from soleved using AI.
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
            
def main():
    try:
        load_dotenv()  # take environment variables from .env
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
        ##initialize our streamlit app
        st.set_page_config(page_title="Advanced Artificial Intelligence Brain")
        st.subheader("Kisan Mitra App")
        st.caption("Developer: Ramendra Singh Rajput.")
        prompt=st.chat_input("Enter Your Question Here")
        lang = st.radio("Select Language:", ("English", "Hindi"))
        with st.sidebar:
         st.write("प्रिय किसान बंधु,") 
         st.write("    मै यहा आपकी किसानी से संबन्धित किसी भी प्रकार की मदद के लिए अग्रसर एक भाषा मॉडल हू जिसे आर्टिफ़िश्यल इंटेलिजेंस वाली मशीन लर्निंग पद्धति से बनाया गया है। आप यहा मुझे अपनी समस्या से अवगत कराएं। मै आपके हर सवाल का जवाब देने की पूरी कोशिश करुगा। मेरे निर्माता द्वारा मुझे निरंतर नयी जानकारियों से प्रशिक्षित कराया जा रहा है। आपसे हुये संवाद से मै निरंतर सीखता जाता हू।")
         st.write("धन्यवाद!")        
        if prompt:
         Kisan_mitra(lang,prompt)
        
    except IOError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
   main()




