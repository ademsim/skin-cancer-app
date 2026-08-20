import os
# Keras için PyTorch backend ayarı
os.environ["KERAS_BACKEND"] = "torch"

import streamlit as st
import gdown
import keras
import numpy as np
from PIL import Image

FILE_ID = '1xbNnagCXfdm0CnYzYWAh3nJLmkOFQkDl'
MODEL_PATH = 'skin_cancer_model.keras'

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            
        with st.spinner("Model buluttan indiriliyor, lütfen bekleyin..."):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
            
    return keras.models.load_model(MODEL_PATH)

# Modeli yükle (Çalışan orijinal yapı)
with st.spinner("Model hazırlanıyor, lütfen bekleyin..."):
    model = load_my_model()

# --- ARAYÜZ VE GÜVENLİ ANALİZ KISMI ---

st.title("Cilt Kanseri Teşhis Asistanı")
st.write("Lütfen analiz etmek istediğiniz cilt lezyonu fotoğrafını yükleyin.")

# Dosya yükleme bileşeni
uploaded_file = st.file_uploader("Bir resim seçin...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Resmi ekranda göster
    image = Image.open(uploaded_file)
    st.image(image, caption="Yüklenen Resim", use_container_width=True)
    
    if st.button("Analiz Et"):
        with st.spinner("Model görüntü üzerinde analiz yapıyor..."):
            try:
                # Şeffaflık (RGBA) veya farklı formatlardaki resimleri standart RGB'ye çevir
                image = image.convert("RGB")
                
                # Modelin ilk eğitim aşamasındaki giriş boyutuna (170x170) göre yeniden boyutlandırıldı[cite: 1]
                img = image.resize((170, 170)) 
                img_array = np.array(img, dtype=np.float32) / 255.0  # Normalizasyon
                
                # Batch boyutu ekle (1, 170, 170, 3)
                img_array = np.expand_dims(img_array, axis=0) 
                
                # Tahmin yapma
                prediction = model.predict(img_array)
                
                # Sonucu ekrana yazdır
                st.success("Analiz Tamamlandı!")
                st.write(f"Tahmin Sonucu: {prediction}")
                
            except Exception as e:
                st.error(f"Analiz sırasında bir hata oluştu: {e}")
