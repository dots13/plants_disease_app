import torch
from torchvision import transforms

import gdown
from pathlib import Path
import streamlit as st
from PIL import Image


import torch.nn as nn

class CoffeeLeafClassifier(nn.Module):
    def __init__(self):
        super(CoffeeLeafClassifier, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 30 * 30, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(128, 5) # 5 classes
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Flatten the output
        x = self.fc_layers(x)
        return x

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Adjust mean and std if necessary
])

model_name = 'coffee_leaf_classifier.pth'
save_dest = Path('models')
save_dest.mkdir(exist_ok=True)
output = f'models/{model_name}'

def load_model_pth(path):
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()
    #model.eval()
    return model
    
def load_model_from_gd():
    f_checkpoint = Path(f"models//{model_name}")
    with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
        gdown.download(id='1XroFNNq4FD8zE3DXDfBPj8NbD7cqYaaf', output=output, quiet=False)
        
# style.py
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
