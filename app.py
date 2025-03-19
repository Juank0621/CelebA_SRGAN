import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from vanillaAE_denoisingAE import VanillaAutoencoder

# # Configuración
# latent_dim = 512
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Carga del modelo
# @st.cache_resource
# def load_model():
#     vanilla_autoencoder = VanillaAutoencoder(latent_dim).to(device)
#     vanilla_autoencoder.load_state_dict(torch.load('vanilla_autoencoder.pth'))
#     vanilla_autoencoder.eval()
#     return vanilla_autoencoder

# vanilla_autoencoder = load_model()

# Interfaz de Streamlit
st.title("Deep Unsupervised Learning - Final Project")
st.sidebar.title("📸 Upload an Image")
uploaded_file = st.sidebar.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
    
#     # Mostrar imagen original
#     st.subheader("Original Image")
#     st.image(image, use_column_width=True)
    
#     # Preprocesamiento
#     preprocess = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor()
#     ])
    
#     input_tensor = preprocess(image).unsqueeze(0).to(device)
    
#     # Forward pass
#     with torch.no_grad():
#         reconstructed = vanilla_autoencoder(input_tensor).cpu().squeeze(0)
    
#     # Mostrar reconstrucción
#     st.subheader("Reconstructed Image")
#     st.image(reconstructed.permute(1, 2, 0).numpy(), use_column_width=True)

#     # Mostrar comparativa lado a lado
#     st.subheader("Comparison")
#     fig, axs = plt.subplots(1, 2, figsize=(8, 4))
#     axs[0].imshow(image)
#     axs[0].set_title("Original")
#     axs[0].axis("off")
#     axs[1].imshow(reconstructed.permute(1, 2, 0).numpy())
#     axs[1].set_title("Reconstructed")
#     axs[1].axis("off")
#     st.pyplot(fig)