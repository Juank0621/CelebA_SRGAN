import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"  # Fix for PyTorch/Streamlit conflict

import streamlit as st
import torch
from PIL import Image
import numpy as np
from src.model import Generator

# Page configuration (must be the first Streamlit command)
st.set_page_config(page_title="SRGAN - Super Resolution", layout="wide")

# Set device: CUDA if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the SRGAN Generator model
@st.cache_resource
def load_generator():
    generator = Generator().to(device)
    generator.load_state_dict(torch.load("models/srgan/srgenerator.pth", map_location=device))
    generator.eval()
    return generator

generator = load_generator()

# Super-resolution function
def super_resolve(image: Image.Image):
    display_size = (256, 256)  # Standardized display size for visualization

    image = image.convert("RGB")
    image_lr = image.resize((64, 64), resample=Image.BICUBIC)

    # Preprocessing
    img_array = np.array(image_lr).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor * 2 - 1  # Normalize to [-1, 1]
    img_tensor = img_tensor.to(device)

    # Inference
    with torch.no_grad():
        sr_tensor = generator(img_tensor).clamp(-1, 1)

    # Postprocessing
    sr_tensor = sr_tensor.cpu()  # Move back to CPU for visualization
    sr_tensor = (sr_tensor + 1) / 2
    sr_image = sr_tensor.squeeze().permute(1, 2, 0).numpy()
    sr_image = Image.fromarray((sr_image * 255).astype(np.uint8))

    return image.resize(display_size), image_lr.resize(display_size), sr_image.resize(display_size)

# Display tech stack badges
def show_badges():
    st.markdown("### 🛠️ Tech Stack")

    cols = st.columns(5)

    with cols[0]:
        st.markdown(
            """
            <a href="https://github.com/Juank0621/CelebA_SRGAN" target="_blank">
                <img src="https://badges.aleen42.com/src/github.svg" alt="GitHub";">
            </a>
            """,
            unsafe_allow_html=True
        )
    with cols[1]:
        st.markdown(
            """
            <a href="https://www.linkedin.com/in/juancarlosgarzon" target="_blank">
                <img src="https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin" alt="LinkedIn";">
            </a>
            """,
            unsafe_allow_html=True
        )
    with cols[2]:
        st.image("https://img.shields.io/badge/Streamlit-1.44.0-orange?logo=streamlit")
    with cols[3]:
        st.image("https://img.shields.io/badge/PyTorch-2.2.0-lightgrey?logo=pytorch")
    with cols[4]:
        st.image("https://img.shields.io/badge/CelebA-dataset-green")

# Main function
def main():
    st.title("📸 SRGAN - Super Resolution on CelebA")
    show_badges()  # Display badges

    st.sidebar.markdown("# 📥 Upload an Image")  # Markdown for larger font (## = H2)
    uploaded_file = st.sidebar.file_uploader("", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)

        st.sidebar.success("Image uploaded successfully.")
        st.sidebar.markdown("Click the button below to run the SRGAN model.")

        if st.sidebar.button("✨ Run SRGAN"):
            original, low_res, super_res = super_resolve(input_image)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(low_res, caption="Low Resolution (64x64)", use_container_width =True)
                
            with col2:
                st.image(super_res, caption="Super Resolution (SRGAN)", use_container_width =True)
            with col3:
                st.image(original, caption="Original Image (256x256)", use_container_width=True)

            st.success(f"Inference completed using `{device}`")
    else:
        st.info("Please upload an image from the sidebar to get started.")

    # Optional footer
    st.markdown("---")
    st.markdown("Made with ❤️ by [Juan Carlos Garzón](https://juancarlosgarzon.com) and [Viviane Alves](https://www.linkedin.com/in/vivialves-developer)")

# Entry point
if __name__ == "__main__":
    main()
