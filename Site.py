import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng as rng

st.set_page_config(layout="wide", page_title="CarView") #Title

st.write("## AutoNerve")
st.write("Upload a video/image file to any information about the cars in the frame such as brand, model, year of release. Additionally, video files will include the timstamp at which the speciifc car appears, and predication confidence.")

st.sidebar.write("## Upload  :gear:")

max_size = 200*1024*1024 #200MB

with st.sidebar:
    option = st.selectbox(
    "Which media format would you like to attach?",
    ("Image", "Video"),
    index=None)
   
    if option == "Image":
        my_upload = st.file_uploader(f"Upload an image", type=["png", "jpg", "jpeg"])
    else:
        my_upload = st.file_uploader("Upload a video", type=["mp4"])

st.markdown(
    f'''
        <style>
            .sidebar .sidebar-content {{
                width: 375px;
            }}
        </style>
    ''',
    unsafe_allow_html=True
)


    
if my_upload is not None: 
    if option == "Image":
        st.image(my_upload.getvalue())
        values = {"2":3}
        timestamp = [2]
        table = pd.DataFrame(values,index = timestamp)
        

        i = 4
        for x in range(1,10):
            table.loc[x] = [i]
            i+=1
            table = pd.DataFrame(values,index = timestamp)
            



    else:
        st.video(my_upload.getvalue())




