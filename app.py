import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

model=load_model('skin_cancer_model.h5')
x=[] #boş liste

def process_image(pil_img):
  img = np.array(pil_img)
  img=cv2.resize(img, (170, 170))  
  img=img/255.0 #normalize et
  img=np.expand_dims(img,axis=0)
  return img

st.title('Deri Kanser resmi sınıflandırma :cancer:')
st.title('Resim seç, model kanser olup olmadığını tahmin etsin!')

file=st.file_uploader('Bir resim yükle',type=['jpg','jpeg','png'])

if file is not None: # Resim yüklenmişse burası çalışacak
  img=Image.open(file)
  st.image(img, caption='Yüklenen Resim')
  image=process_image(img)
  prediction=model.predict(image)
  predicted_class= 1 if prediction > 0.5 else 0
  class_names=['Kanser Değil','Kanser']
  st.write(class_names[predicted_class])
