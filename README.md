<p align="center">
  <img src="images/personal_logo.png" alt="Logo" width="400"/>
</p>

# 🧠 Generative AI + Anomaly Detection on CIFAR-10 🖼️
<p align="center">Developed by <a href="https://www.linkedin.com/in/juancarlosgarzon/">Juan Carlos Garzon</a>, <a href="https://www.linkedin.com/in/vivialves-developer/">Viviane Alves</a> </p>

---

<p align="center">
  <img src="cifar10.png" alt="Image" width="600"/>
</p>

---

## 🎥 Demo VIDEO
*(Insert link to video demo here)*

---

## 📚 Dataset

The project uses the **CIFAR-10** dataset, containing 60,000 32x32 color images across 10 distinct classes (6,000 images per class):

- ✈️ **Airplane**: 6,000 images
- 🚗 **Automobile**: 6,000 images
- 🐦 **Bird**: 6,000 images
- 🐱 **Cat**: 6,000 images
- 🦌 **Deer**: 6,000 images
- 🐶 **Dog**: 6,000 images
- 🐸 **Frog**: 6,000 images
- 🐴 **Horse**: 6,000 images
- 🚢 **Ship**: 6,000 images
- 🚚 **Truck**: 6,000 images

---

## 🎯 Project Goal

This project implements **Generative AI models** (Autoencoders, Variational Autoencoders, GANs) on **CIFAR-10**, applying them for **Anomaly Detection**.

After training, a **Streamlit** web app allows users to:
- Visualize image reconstructions
- Generate synthetic samples
- Detect anomalies based on reconstruction error or GAN discriminator scores

---

## 🛠️ Key Components

- **Dataset**: CIFAR-10
- **Models**:
  - Autoencoder
  - Variational Autoencoder (VAE)
  - Generative Adversarial Network (GAN)
- **Anomaly Detection**: Based on reconstruction loss or GAN scores
- **Web Interface**: Built with Streamlit

---

## 🔍 CIFAR-10 Classes
- ✈️ Airplane
- 🚗 Automobile
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐶 Dog
- 🐸 Frog
- 🐴 Horse
- 🚢 Ship
- 🚚 Truck

---

## 🚀 Getting Started

To reproduce this project, follow these steps. The project was trained on **Ubuntu** using an **NVIDIA RTX 4080 GPU**.

### Prerequisites
- **Miniforge/Mamba** (environment management)
- **Visual Studio Code** (or other IDE)
- **NVIDIA Drivers + CUDA**
