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

model_path = os.path.join('cnn_strategy1_weighted_loss', 'coffee_leaf_classifier.pth')

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