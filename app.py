import os
os.environ["KERAS_BACKEND"] = "torch"

import streamlit as st
import gdown
import keras

FILE_ID = 'https://drive.google.com/file/d/1xbNnagCXfdm0CnYzYWAh3nJLmkOFQkDl/view?usp=sharing'
MODEL_PATH = 'skin_cancer_model.h5' # veya .keras

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Model buluttan indiriliyor, lütfen bekleyin..."):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
            
    # Modern Keras 3 ile yükleme
    return keras.models.load_model(MODEL_PATH)

model = load_my_model()
