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

def get_gemini_response(prompt):
    model = GenerativeModel('gemini-pro')
    response = model.generate_content([prompt])
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

def Kisan_mitra1():
    
    input_prompt = """
                   You are an agriculture expert in hindi language. Expert in Answering questions of farmers related to agriculture in hindi language.Here you have to generate a question on behalf of former regarding the crop disease and answer regarding this disease in hindi language. You are trained by Ramendra Singh Rajput, working for Mp govt as a patwari since 2015 and have gained greate experience to work with land records, citizence problem solving, land measorment, managing the data of citizence, providing them end to end goverment services in variouse manners. Have gained experience of different different fields work provided by goverment in line order duties. Utilizing this knowledge to develop a powerfull echo system for goverment to help people and solve theire problem in a smart way. He is an Artificial intelligence expert, Machine learning and Deep learning engineer,also working on Health Expert System, Music Expert System projects.He is having google developer profile.His education and qualification is Bachelore of computer application from M.I.M.T. college Narsimhapur(2007-2010),  master of computer application from ShriRam Institue Of Technology and Science(2010 to 2012).Active learner for Machine learning, Deep learning and Generative AI.Keen in making corelation between phylosophy and quantom physics.His email id is ramendra.rajput85@gmail.com, linkedin id is https://www.linkedin.com/in/ramendra-singh-rajput-026a6a22/ , Google developer profile is https://g.dev/ramendrarajput, live projects link: https://advanced-artificial-intelligence-brain.streamlit.app/, Github link for projects: https://github.com/ramendrarajput/ ,
                   you will have to put all links in bottom of page and each link should be seperated by a new line. Have a notification at bottom to contact me to get there business from solved using AI.
                   """

    response = get_gemini_response(input_prompt)
    if response:
     st.success('Done')
     st.write(response)
   

def main():
    try:
        load_dotenv()  # take environment variables from .env
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
        ##initialize our streamlit app
        st.set_page_config(page_title="Advanced Artificial Intelligence Brain",page_icon="Kisan-Mitra.png")
        #sidebar = st.sidebar(expanded=True)
        st.subheader("किसान मित्र चैटबॉट")
        st.caption("Developer: Ramendra Singh Rajput")
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
            
        
    except IOError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
   main()




