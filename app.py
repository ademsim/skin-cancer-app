import os
os.environ["KERAS_BACKEND"] = "torch"

import streamlit as st
import urllib.request
import keras

FILE_ID = '1xbNnagCXfdm0CnYzYWAh3nJLmkOFQkDl'
MODEL_PATH = 'skin_cancer_model.keras' 

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Model downloading, please wait..."):
            # bypass G-Drive for large file
            url = f'https://drive.google.com/uc?export=download&id={FILE_ID}&confirm=t'
            urllib.request.urlretrieve(url, MODEL_PATH)
            
    return keras.models.load_model(MODEL_PATH)

model = load_my_model()
