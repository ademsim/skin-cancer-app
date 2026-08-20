import os
os.environ["KERAS_BACKEND"] = "torch"

import streamlit as st
import gdown
import keras

FILE_ID = '1xbNnagCXfdm0CnYzYWAh3nJLmkOFQkDl'
MODEL_PATH = 'skin_cancer_model.keras'

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            
        with st.spinner("Model buluttan indiriliyor, lütfen bekleyin..."):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            # Hiçbir ek parametre vermeden en sade ve kararlı çağrı
            gdown.download(url, MODEL_PATH, quiet=False)
            
    return keras.models.load_model(MODEL_PATH)

model = load_my_model()
