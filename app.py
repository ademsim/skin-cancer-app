import os
os.environ["KERAS_BACKEND"] = "torch"

import streamlit as st
import gdown
import keras

FILE_ID = 'https://drive.google.com/file/d/1xbNnagCXfdm0CnYzYWAh3nJLmkOFQkDl/view?usp=sharing'
MODEL_PATH = 'skin_cancer_model.keras'

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Model downloading, please wait..."):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
            
    # Keras-loading function
    return keras.models.load_model(MODEL_PATH)

model = load_my_model()
