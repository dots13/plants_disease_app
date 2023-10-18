import torch
from torchvision import transforms

import gdown
from pathlib import Path
import streamlit as st
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Adjust mean and std if necessary
])

model_name = 'coffee_leaf_classifier.pth'
save_dest = Path('models')
save_dest.mkdir(exist_ok=True)
output = f'models/{model_name}'

@st.cache_resource
def load_model_pth(path):
    return torch.load(path)
    
def load_model_from_gd():
    f_checkpoint = Path(f"models//{model_name}")
    with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
        gdown.download(id='1XroFNNq4FD8zE3DXDfBPj8NbD7cqYaaf', output=output, quiet=False)
        
        
f_checkpoint = Path(f"models//{model_name}")
if not f_checkpoint.exists():
    load_model_from_gd()
else:
    modelicka = load_model_pth(f_checkpoint)
    
    
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
    # Apply the transformations to the image
    image_tensor = transform(image)

    # Add a batch dimension (since models expect a batch of images, not a single image)
    image_tensor = image_tensor.unsqueeze(0)

    # Move to the same device as the model (if using CUDA)
    image_tensor = image_tensor.to(device)

    # Pass the image through the model
    output = model(image_tensor)

    # Get the predicted class
    _, predicted_class = torch.max(output, 1)
    st.write(predicted_class)
