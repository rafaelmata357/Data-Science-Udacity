import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image


st.title('Introduction to building Streamlit WebApp')
st.sidebar.title('This is the sidebar')
st.sidebar.markdown('Let’s start with binary classification!!')

img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])


#st.balloons()

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    st.image(
    image, caption=f"Processed image", width= 400) #use_column_width=True)

basewidth = 100
#st.write(str(image.size[0]))

#wpercent = (basewidth/float(image.size[0]))
#hsize = int((float(image.size[1])*float(wpercent)))
#image = image.resize((basewidth,hsize), Image.ANTIALIAS)






st.subheader('Support Vector Machine (SVM) results')