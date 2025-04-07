import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchsummary import summary
from torchvision.models import resnet18  # Import ResNet18
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # Import for confusion matrix

torch.set_float32_matmul_precision('medium')

# CIFAR10 images are already 32x32, so no resizing is needed. 
# We normalize the images to have a mean of 0.5 and a standard deviation of 0.5 for each channel.
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR10 mean/std
])

kwargs = {'num_workers': 8, 'pin_memory': True}  # DataLoader optimization for better performance on CUDA.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom dataset class to filter CIFAR10 images by a specific class.
class OneClassDatasetCIFAR10(CIFAR10):
    def __init__(self, root_dir, real_class=1, transform=None, train=True, download=True):
        super().__init__(root=root_dir, transform=transform, train=train, download=download)
        self.real_class = real_class
        self.samples = []
        for i in range(len(self.data)):
            if self.targets[i] == self.real_class:
                self.samples.append((self.data[i], self.targets[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        image = TF.to_tensor(data[0])
        if self.transform:
            image = self.transform(image)
        label = 0  # Dummy label since the autoencoder does not use labels.
        return image, label

# Encoder for the VAE using ResNet18 architecture
class VAE_Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE_Encoder, self).__init__()
        # Load ResNet18 without pre-trained weights
        resnet = resnet18(weights=None)
        # Remove the fully connected layer (fc) to use it as a feature extractor
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last fc layer
        self.flatten = nn.Flatten()  # Flatten the output of the feature extractor
        self.fc_mean = nn.Linear(512, latent_dim)  # Map to latent space (mean)
        self.fc_logvar = nn.Linear(512, latent_dim)  # Map to latent space (log variance)

    def forward(self, x):
        x = self.feature_extractor(x)  # Extract features
        x = self.flatten(x)  # Flatten the features
        mean = self.fc_mean(x)  # Compute mean
        logvar = self.fc_logvar(x)  # Compute log variance
        return mean, logvar

# Decoder for the VAE using transposed convolutional layers. It reconstructs the image from the latent space
# back to the original 32x32 resolution with 3 channels.
class VAE_Decoder(nn.Module):
    def __init__(self, latent_dim=256):  # Adjust latent space size
        super(VAE_Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 2048)  # Increase decoder capacity
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, 2, 2)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = self.decoder(x)
        return x

# Variational Autoencoder (VAE) combining the encoder and decoder. It includes the reparameterization trick
# to sample from the latent space.
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(latent_dim)
        self.decoder = VAE_Decoder(latent_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decoder(z)
        return x_hat, mean, logvar

# Loss function for the VAE
def vae_loss_function(x, x_hat, mean, log_var, beta=0.1):
    reconstruction_loss = nn.functional.mse_loss(x_hat, x, reduction='mean')  # Reconstruction loss
    kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())  # KL divergence
    kl_divergence = torch.clamp(kl_divergence, min=-1e6, max=1e6)  # Clamp to avoid numerical issues
    return reconstruction_loss + beta * kl_divergence / x.size(0)  # Normalize KL by batch size

# Function to train the VAE for one epoch with tqdm
def train_vae_epoch(vae, train_loader, optimizer):
    vae.train()
    total_loss = 0
    # Use tqdm to display progress
    for inputs, _ in tqdm(train_loader, desc="Training VAE Epoch", leave=False):
        inputs = inputs.to(device)  # Move inputs to device
        inputs = torch.clamp(inputs, 0., 1.)  # Normalize inputs to range [0, 1]
        optimizer.zero_grad()
        x_hat, mean, log_var = vae(inputs)
        loss = vae_loss_function(inputs, x_hat, mean, log_var, beta=0.1)
        if torch.isnan(loss):
            print("NaN detected in loss! Skipping batch.")
            continue
        loss.backward()
        
        # Log gradients for debugging
        for name, param in vae.named_parameters():
            if param.grad is not None:
                print(f"Gradient for {name}: {param.grad.norm().item()}")

        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Function to evaluate the VAE on the validation set
def evaluate_vae(vae, val_loader):
    vae.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)  # Move inputs to device
            x_hat, mean, log_var = vae(inputs)
            loss = vae_loss_function(inputs, x_hat, mean, log_var)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# Function to test the VAE for anomaly detection
def test_vae(vae, test_loader, threshold):
    vae.eval()
    anomalies = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)  # Move inputs to device
            x_hat, mean, log_var = vae(inputs)
            z = vae.reparameterize(mean, log_var)
            distances = [calculate_mahalanobis_distance(mean[i], torch.eye(mean.size(1)).to(device), z[i]) for i in range(z.size(0))]
            anomalies.extend([d > threshold for d in distances])
    return anomalies

# Function to calculate Mahalanobis distance
def calculate_mahalanobis_distance(mean, covariance, x):
    diff = x - mean
    inv_covariance = torch.linalg.inv(covariance)
    distance = torch.sqrt(torch.mm(torch.mm(diff.unsqueeze(0), inv_covariance), diff.unsqueeze(1)))
    return distance.item()

# Main function to train the VAE for all classes
def main_vae():
    latent_dim = 256  # Latent space size
    num_epochs = 100  # Number of epochs
    learning_rate = 1e-3

    vae = VAE(latent_dim).to(device)  # Initialize VAE

    # Display the architecture of the model
    summary(vae, input_size=(3, 32, 32))  # Input size matches CIFAR10 images

    optimizer = torch.optim.AdamW(vae.parameters(), lr=learning_rate)  # Use AdamW optimizer

    all_train_losses = []
    all_val_losses = []

    for real_class in range(10):
        print(f"Training VAE for class {real_class}...")

        train_dataset = OneClassDatasetCIFAR10(root_dir='data', real_class=real_class, transform=transform, train=True, download=True)
        val_dataset = OneClassDatasetCIFAR10(root_dir='data', real_class=real_class, transform=transform, train=False, download=True)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, **kwargs)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, **kwargs)

        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            train_loss = train_vae_epoch(vae, train_loader, optimizer)
            val_loss = evaluate_vae(vae, val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        # Save the trained model
        torch.save(vae.state_dict(), f'models/vae/vae_class_{real_class}_weights.pth')

    return all_train_losses, all_val_losses


# Call the main function for VAE
all_train_losses, all_val_losses = main_vae()

# Example usage of test_vae
def main_vae_with_test():
    latent_dim = 256  # Latent space size
    num_epochs = 100  # Number of epochs
    learning_rate = 1e-3
    threshold = 5.0  # Example threshold for Mahalanobis distance

    vae = VAE(latent_dim).to(device)  # Initialize VAE

    # Display the architecture of the model
    summary(vae, input_size=(3, 32, 32))  # Input size matches CIFAR10 images

    optimizer = torch.optim.AdamW(vae.parameters(), lr=learning_rate)  # Use AdamW optimizer

    all_train_losses = []
    all_val_losses = []

    for real_class in range(10):
        print(f"Training VAE for class {real_class}...")

        train_dataset = OneClassDatasetCIFAR10(root_dir='data', real_class=real_class, transform=transform, train=True, download=True)
        val_dataset = OneClassDatasetCIFAR10(root_dir='data', real_class=real_class, transform=transform, train=False, download=True)
        test_dataset = CIFAR10(root='data', train=False, transform=transform, download=True)  # Full test dataset

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, **kwargs)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, **kwargs)

        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            train_loss = train_vae_epoch(vae, train_loader, optimizer)
            val_loss = evaluate_vae(vae, val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        # Save the trained model
        torch.save(vae.state_dict(), f'models/vae/vae_class_{real_class}_weights.pth')

        # Test the VAE on the test dataset
        print(f"Testing VAE for class {real_class}...")
        anomalies = test_vae(vae, test_loader, threshold)
        print(f"Anomalies detected for class {real_class}: {sum(anomalies)} out of {len(anomalies)}")

    return all_train_losses, all_val_losses

# Call the main function with testing
all_train_losses, all_val_losses = main_vae_with_test()

# Function to plot training and validation losses for each class in separate plots
def plot_losses_per_class_separately(all_train_losses, all_val_losses):
    num_classes = len(all_train_losses)
    for class_index in range(num_classes):
        plt.figure(figsize=(10, 5))
        plt.plot(all_train_losses[class_index], label='Train Loss')
        plt.plot(all_val_losses[class_index], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Losses for Class {class_index}')
        plt.legend()
        plt.show()

# Generate separate plots for losses after training
plot_losses_per_class_separately(all_train_losses, all_val_losses)

# Function to visualize the reconstruction of an image by the model
def visualize_reconstruction(vae, dataset, class_index):
    vae.eval()
    with torch.no_grad():
        # Select a random image from the dataset of the given class
        image, _ = dataset[np.random.randint(len(dataset))]
        image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
        reconstructed, _, _ = vae(image)

        # Convert tensors to numpy arrays for visualization
        original_image = image.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5  # Denormalize
        reconstructed_image = reconstructed.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5  # Denormalize

        # Plot original and reconstructed images
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title(f'Original Image (Class {class_index})')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed_image)
        plt.title(f'Reconstructed Image (Class {class_index})')
        plt.axis('off')

        plt.show()



# Visualize reconstruction for each class
for class_index in range(10):
    print(f"Visualizing reconstruction for class {class_index}...")
    # Load the corresponding dataset and model for the class
    val_dataset = OneClassDatasetCIFAR10(root_dir='data', real_class=class_index, transform=transform, train=False, download=True)
    vae = VAE(latent_dim=128).to(device)
    vae.load_state_dict(torch.load(f'models/vae/vae_class_{class_index}_weights.pth'))
    visualize_reconstruction(vae, val_dataset, class_index)

# Function to generate a confusion matrix for anomaly detection
def generate_confusion_matrix(vae, real_class, threshold, transform):
    vae.eval()
    all_labels = []
    all_predictions = []

    # Load datasets for the real class and other classes
    real_dataset = OneClassDatasetCIFAR10(root_dir='data', real_class=real_class, transform=transform, train=False, download=True)
    other_dataset = [OneClassDatasetCIFAR10(root_dir='data', real_class=i, transform=transform, train=False, download=True)
                     for i in range(10) if i != real_class]

    # Evaluate real class (normal samples)
    with torch.no_grad():
        for image, _ in real_dataset:
            image = image.unsqueeze(0).to(device)
            x_hat, mean, log_var = vae(image)
            z = vae.reparameterize(mean, log_var)
            distance = calculate_mahalanobis_distance(mean[0], torch.eye(mean.size(1)).to(device), z[0])
            all_labels.append(0)  # Label 0 for normal
            all_predictions.append(1 if distance > threshold else 0)

    # Evaluate other classes (anomalous samples)
    with torch.no_grad():
        for dataset in other_dataset:
            for image, _ in dataset:
                image = image.unsqueeze(0).to(device)
                x_hat, mean, log_var = vae(image)
                z = vae.reparameterize(mean, log_var)
                distance = calculate_mahalanobis_distance(mean[0], torch.eye(mean.size(1)).to(device), z[0])
                all_labels.append(1)  # Label 1 for anomaly
                all_predictions.append(1 if distance > threshold else 0)

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for Class {real_class}")
    plt.show()

# Generate confusion matrices for all classes
threshold = 5.0  # Example threshold for Mahalanobis distance
for class_index in range(10):
    print(f"Generating confusion matrix for class {class_index}...")
    vae = VAE(latent_dim=128).to(device)
    vae.load_state_dict(torch.load(f'models/vae/vae_class_{class_index}_weights.pth'))
    generate_confusion_matrix(vae, class_index, threshold, transform)