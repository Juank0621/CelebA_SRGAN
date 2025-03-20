import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from model import VanillaAutoencoder

# Configuration
latent_dim = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
@st.cache_resource
def load_model():
    vanilla_autoencoder = VanillaAutoencoder(latent_dim).to(device)
    vanilla_autoencoder.load_state_dict(torch.load('vanilla_autoencoder.pth'))
    vanilla_autoencoder.eval()
    return vanilla_autoencoder

# Main Streamlit app function
def main():
    # Page title
    st.title("🧒🏻 Deep Unsupervised Learning - Final Project 👧🏼")
    st.subheader("🚀 Juan Carlos Garzon - Viviane Alves 🚀")
    st.sidebar.title("📸 Upload Files")

    # File uploader in the sidebar
    uploaded_file = st.sidebar.file_uploader("📤Upload an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Display original image
        st.subheader("🖼️ Original Image")
        st.image(image, use_container_width=True)

        # Preprocessing steps for the image
        preprocess = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        
        input_tensor = preprocess(image).unsqueeze(0).to(device)

        # Load the model only once
        vanilla_autoencoder = load_model()

        # Forward pass through the model
        with torch.no_grad():
            reconstructed = vanilla_autoencoder(input_tensor).cpu().squeeze(0)

        # Display reconstructed image
        st.subheader("🖼️ Reconstructed Image")
        st.image(reconstructed.permute(1, 2, 0).numpy(), use_container_width=True)

        # Display comparison between original and reconstructed images side by side
        st.subheader("🎯 Comparison")
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(image)
        axs[0].set_title("Original")
        axs[0].axis("off")
        axs[1].imshow(reconstructed.permute(1, 2, 0).numpy())
        axs[1].set_title("Reconstructed")
        axs[1].axis("off")
        st.pyplot(fig)

if __name__ == "__main__":
    main()