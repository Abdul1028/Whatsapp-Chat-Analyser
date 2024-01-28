import streamlit as st
from streamlit_option_menu import option_menu
from app_utils import switch_page
import streamlit as st
from PIL import Image
import openai
import os

im = Image.open("icon.png")
openai.api_key = os.getenv("OPENAI_API_KEY")
st.set_page_config(page_title = "AI Interviewer", layout = "centered",page_icon=im)

home_title = "AI Interviewer"
home_introduction = "Welcome to AI Interviewer, empowering your interview preparation with generative AI."
with st.sidebar:
    st.markdown('AI Interviewer')
    st.markdown(""" 
       #### Powered and Enhanced by

       [OpenAI](https://openai.com/)

       [Langchain](https://github.com/hwchase17/langchain)
       
       [Abdul intelligence](https://github.com/Abdul1028)

                   """)
st.markdown(
    "<style>#MainMenu{visibility:hidden;}</style>",
    unsafe_allow_html=True
)
st.image(im, width=100)
st.markdown(f"""# {home_title} <span style=color:#2E9BF5><font size=5>Beta</font></span>""", unsafe_allow_html=True)
st.markdown("""\n""")
# st.markdown("#### Greetings")
st.markdown(
    "Welcome to AI Interviewer! 👏 AI Interviewer is your personal interviewer powered by generative AI that conducts mock interviews."
    "You can upload your resume and enter job descriptions, and AI Interviewer will ask you customized questions. Additionally, you can configure your own Interviewer!")
st.markdown("""\n""")
with st.expander("What's coming next?"):
    st.write("""
       Video conferencing Interviews """)
st.markdown("""\n""")
st.markdown("#### Get started!")
st.markdown("Select one of the following screens to start your interview!")
selected = option_menu(
    menu_title=None,
    options=["Professional", "Resume", "Behavioral", "Customize!"],
    icons=["cast", "cloud-upload", "cast"],
    default_index=0,
    orientation="horizontal",
)
if selected == 'Professional':
    st.info("""
           📚In this session, the AI Interviewer will assess your technical skills as they relate to the job description.
           Note: The maximum length of your answer is 4097 tokens!
           - Each Interview will take 10 to 15 mins.
           - To start a new session, just refresh the page.
           - Start introduce yourself and enjoy！ """)
    if st.button("Start Interview!"):
        switch_page("Professional Screen")
if selected == 'Resume':
    st.info("""
       📚In this session, the AI Interviewer will review your resume and discuss your past experiences.
       Note: The maximum length of your answer is 4097 tokens!
       - Each Interview will take 10 to 15 mins.
       - To start a new session, just refresh the page.
       - Start introduce yourself and enjoy！ """
            )
    if st.button("Start Interview!"):
        switch_page("Resume Screen")
if selected == 'Behavioral':
    st.info("""
       📚In this session, the AI Interviewer will assess your soft skills as they relate to the job description.
       Note: The maximum length of your answer is 4097 tokens!
       - Each Interview will take 10 to 15 mins.
       - To start a new session, just refresh the page.
       - Start introduce yourself and enjoy！ 
       """)
    if st.button("Start Interview!"):
        switch_page("Behavioral Screen")
if selected == 'Customize!':
    st.info("""
           📚In this session, you can customize your own AI Interviewer and practice with it!
            - Coming soon

            Coming at the end of TY BSC CS""")
st.markdown("""\n""")

