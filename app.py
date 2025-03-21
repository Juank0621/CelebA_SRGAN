import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from model import VanillaAutoencoder
import os

# Configuration
latent_dim = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
@st.cache_resource
def load_model(model_type, class_name):
    model_path = f'models/{model_type}/{model_type}_class_{class_name}_weights.pth'
    autoencoder = VanillaAutoencoder(latent_dim).to(device)
    autoencoder.load_state_dict(torch.load(model_path))
    autoencoder.eval()
    return autoencoder

# Main Streamlit app function
def main():
    # Page title
    st.title("🧒🏻 Deep Unsupervised Learning - Final Project 👧🏼")
    st.subheader("🚀 Juan Carlos Garzon - Viviane Alves 🚀")
    st.sidebar.title("📸 Upload Files")

    # Dropdown to select the model type
    model_types = ['vanilla_autoencoder', 'denoising_autoencoder']
    model_display_names = [model_type.replace('_', ' ') for model_type in model_types]
    selected_model_display_name = st.sidebar.selectbox("Select Model Type", model_display_names)
    selected_model_type = model_types[model_display_names.index(selected_model_display_name)]

    # Dropdown to select the class model
    class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    selected_class = st.sidebar.selectbox("Select Class Model", class_names)
    class_index = class_names.index(selected_class)

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
        autoencoder = load_model(selected_model_type, class_index)

        # Forward pass through the model
        with torch.no_grad():
            reconstructed = autoencoder(input_tensor).cpu().squeeze(0)

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

        # Calculate MSE loss to determine if the image is an anomaly
        mse_loss = torch.nn.functional.mse_loss(reconstructed, input_tensor.cpu().squeeze(0), reduction='mean').item()

        if selected_model_type == 'vanilla_autoencoder':
            if class_index == 0:
                threshold = 0.0010
            else:
                threshold = 0.1109
        elif selected_model_type == 'denoising_autoencoder':
            if class_index == 0:
                threshold = 0.9900
            elif class_index == 1 or class_index == 5 or class_index == 8:
                threshold = 0.2208
            elif class_index == 2 or class_index == 6:
                threshold = 0.7702
            elif class_index == 3 or class_index == 4:
                threshold = 0.4406
            elif class_index == 7 or class_index == 9:
                threshold = 0.1109
        print(class_index)
        print(selected_model_type)
        print(threshold)
        print(mse_loss)
        # threshold = 0.01  # This threshold should be determined based on your training results

        # Display anomaly detection result
        if mse_loss > threshold:
            st.error("🚨 This image is an anomaly!")
        else:
            st.success("✅ This image is normal.")

if __name__ == "__main__":
    main()