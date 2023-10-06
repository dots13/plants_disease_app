import tensorflow as tf
from tensorflow.keras.models import load_model
import gdown
from pathlib import Path
import streamlit as st
from PIL import Image

st.set_page_config(page_title='Plants diseases', page_icon=':herb:', initial_sidebar_state='auto')

model_name = 'withouth_cersc_resnet50_deduplicated_mix_val_train_75acc.h5'

save_dest = Path('models')
save_dest.mkdir(exist_ok=True)
output = f'models/{model_name}'

@st.cache_resource
def load_model_h5(path):
    return load_model(output, compile=False)

def load_model_from_gd():
    #f_checkpoint = Path(f"models//{model_name}")
    with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
        gdown.download(id='1--eYkRRQl6CAuXxPFcgiFy0zdp67WTPE', output=output, quiet=False)
        
f_checkpoint = Path(f"models//{model_name}")
if not f_checkpoint.exists():
    load_model_from_gd()
else:
    modelicka = load_model_h5(f_checkpoint)
    
uploaded_file = st.file_uploader("Upload file", ["png", "jpg"], key='uploader')

if uploaded_file:
    show_file = st.empty()
    show_file.image(uploaded_file)
   
if st.session_state.get("uploader", False):
    st.session_state.disabled = False
else:
    st.session_state.disabled = True
    
classify_button = st.button("Classify", key='c_but', disabled=st.session_state.get("disabled", True))

DEFAULT_IMAGE_SIZE = tuple((224, 224))
if classify_button:
    image = Image.open(uploaded_file)
    image = image.resize(DEFAULT_IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    res = modelicka.predict(img_array)
    st.write(res)


    
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
