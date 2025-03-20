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

from sklearn.metrics import roc_curve, auc

torch.set_float32_matmul_precision('medium')

# Define transformations for the CIFAR10 dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
]) 

# Custom Dataset class for loading CIFAR10 images of a specific class
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
        image = image / 255

        if self.transform:
            image = self.transform(image)

        label = 0  # Dummy label as autoencoder does not need labels

        return image, label

# Load the CIFAR10 dataset
train_dataset = CIFAR10(root='data', train=True, download=True, transform=None)
test_dataset = CIFAR10(root='data', train=False, download=True, transform=None)

# Split the training dataset into training and validation sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Verify the number of images in each dataset
print(f'Number of images in training dataset: {len(train_dataset)}')
print(f'Number of images in validation dataset: {len(val_dataset)}')
print(f'Number of images in test dataset: {len(test_dataset)}')

# Create data loaders for the training, validation, and test datasets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Function to show images from the CelebA dataset
def show_transform_images(loader):
    data_iter = iter(loader)
    images, _ = next(data_iter)
    images = images.numpy()
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i in range(2):
        for j in range(5):
            axes[i, j].imshow(np.transpose(images[i * 5 + j], (1, 2, 0)))
            axes[i, j].set_xticks(np.arange(0, images.shape[2], 32))
            axes[i, j].set_yticks(np.arange(0, images.shape[3], 32))
    plt.show()

# Show images from the CelebA dataset
show_transform_images(train_loader)

# Define the Vanilla Autoencoder Model
# The latent_dim variable defines the size of the latent space.
latent_dim = 512

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),  # 128x128 → 64x64
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),  # 64x64 → 32x32
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(2),  # 32x32 → 16x16
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.MaxPool2d(2),  # 16x16 → 8x8
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.MaxPool2d(2),  # 8x8 → 4x4
            
            nn.Flatten(),  # Flatten to a 1D vector
            nn.Linear(4 * 4 * 512, latent_dim),  # Match the output to the latent dimension
            nn.SiLU()
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 512),
            nn.SiLU(),
            nn.Unflatten(1, (512, 4, 4)),

            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4x4 → 8x8
            nn.BatchNorm2d(256),
            nn.SiLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8 → 16x16
            nn.BatchNorm2d(128),
            nn.SiLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16 → 32x32
            nn.BatchNorm2d(64),
            nn.SiLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32 → 64x64
            nn.BatchNorm2d(32),
            nn.SiLU(),

            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x64 → 128x128
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)

class VanillaAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(VanillaAutoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

vanilla_autoencoder = VanillaAutoencoder(latent_dim)

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vanilla_autoencoder.to(device)

summary(vanilla_autoencoder, input_size=(3, 128, 128))

# Function to train the model for one epoch
def train_epoch(vanilla_autoencoder, train_loader, criterion, optimizer):
    vanilla_autoencoder.train()
    total_loss = 0
    for inputs, _ in train_loader:
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = vanilla_autoencoder(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Function to evaluate the model
def evaluate_model(vanilla_autoencoder, val_loader, criterion):
    vanilla_autoencoder.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            outputs = vanilla_autoencoder(inputs)
            loss = criterion(outputs, inputs)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# Function to test the model and calculate AUROC
def test_model(vanilla_autoencoder, real_test_loader, ano_test_loaders):
    vanilla_autoencoder.eval()

    def get_score(loader):
        log_probs = []
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(device)
                outputs = vanilla_autoencoder(inputs)
                mse_loss = F.mse_loss(outputs, inputs, reduction='none').mean(dim=[1, 2, 3])
                log_probs.append(mse_loss.cpu().numpy())
        return np.concatenate(log_probs)

    real_log_probs = get_score(real_test_loader)
    ano_log_probs = [get_score(loader) for loader in ano_test_loaders]
    aurocs = []
    for ano_log_prob in ano_log_probs:
        y_true = np.concatenate([np.ones_like(real_log_probs), np.zeros_like(ano_log_prob)])
        y_score = np.concatenate([real_log_probs, ano_log_prob])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        aurocs.append(roc_auc)

    return np.mean(aurocs)

# Function to select the threshold
def select_threshold(vanilla_autoencoder, real_test_loader, ano_test_loader):
    vanilla_autoencoder.eval()

    def get_score(loader):
        log_probs = []
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(device)
                outputs = vanilla_autoencoder(inputs)
                mse_loss = F.mse_loss(outputs, inputs, reduction='none').mean(dim=[1, 2, 3])
                log_probs.append(mse_loss.cpu().numpy())
        return np.concatenate(log_probs)

    real_log_probs = get_score(real_test_loader)
    ano_log_probs = get_score(ano_test_loader)
    y_true = np.concatenate([np.ones_like(real_log_probs), np.zeros_like(ano_log_probs)])
    y_score = np.concatenate([real_log_probs, ano_log_probs])

    thresholds = np.linspace(0.001, 0.99, 10)
    best_threshold = 0
    best_accuracy = 0
    for threshold in thresholds:
        binary_score = y_score >= threshold
        correct = (binary_score == y_true).sum().item()
        acc = correct / len(y_true)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = threshold

    return best_threshold

def save_weights(model, path):
    torch.save(model.state_dict(), path)

# Function to display 5 original and reconstructed images from the test dataset
def display_test_images(vanilla_autoencoder, test_loader):
    vanilla_autoencoder.eval()
    images, reconstructed_images = [], []
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= 5:
                break
            inputs = inputs.to(device)
            outputs = vanilla_autoencoder(inputs)
            images.append(inputs.cpu().squeeze(0))
            reconstructed_images.append(outputs.cpu().squeeze(0))

    fig, axs = plt.subplots(5, 2, figsize=(10, 20))
    for i in range(5):
        axs[i, 0].imshow(images[i].permute(1, 2, 0).numpy())
        axs[i, 0].set_title("Original")
        axs[i, 0].axis("off")
        axs[i, 1].imshow(reconstructed_images[i].permute(1, 2, 0).numpy())
        axs[i, 1].set_title("Reconstructed")
        axs[i, 1].axis("off")
    plt.show()

# Main function to train and evaluate the model
def main():
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize confusion matrix and lists to store results
    confusion_mat = np.zeros((10, 10))
    all_aurocs = []
    all_thresholds = []
    all_train_losses = []
    all_val_losses = []

    # Loop over each class as the real class
    for real_class in range(10):
        print(f'Training for real class: {real_class}')

        # Load datasets
        train_dataset = OneClassDatasetCIFAR10(root_dir='data', real_class=real_class, transform=transform, train=True, download=True)
        val_dataset = OneClassDatasetCIFAR10(root_dir='data', real_class=real_class, transform=transform, train=False, download=True)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

        real_test_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        ano_test_loaders = [DataLoader(OneClassDatasetCIFAR10(root_dir='data', real_class=i, transform=transform, train=False, download=True), batch_size=32, shuffle=False) for i in range(10) if i != real_class]

        # Initialize model, criterion, and optimizer
        vanilla_autoencoder = VanillaAutoencoder(latent_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(vanilla_autoencoder.parameters(), lr=0.0005)

        # Training loop
        num_epochs = 20
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            train_loss = train_epoch(vanilla_autoencoder, train_loader, criterion, optimizer)
            val_loss = evaluate_model(vanilla_autoencoder, val_loader, criterion)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Store losses
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        # Calculate AUROC
        auroc = test_model(vanilla_autoencoder, real_test_loader, ano_test_loaders)
        all_aurocs.append(auroc)
        print(f'AUROC for class {real_class}: {auroc:.4f}')

        # Select threshold
        threshold = select_threshold(vanilla_autoencoder, real_test_loader, ano_test_loaders[0])
        all_thresholds.append(threshold)
        print(f'Selected Threshold for class {real_class}: {threshold:.4f}')

        # Update confusion matrix
        def get_predictions(loader, threshold):
            predictions = []
            with torch.no_grad():
                for inputs, _ in loader:
                    inputs = inputs.to(device)
                    outputs = vanilla_autoencoder(inputs)
                    mse_loss = F.mse_loss(outputs, inputs, reduction='none').mean(dim=[1, 2, 3])
                    preds = (mse_loss < threshold).cpu().numpy()
                    predictions.append(preds)
            return np.concatenate(predictions)

        real_preds = get_predictions(real_test_loader, threshold)
        for i, ano_loader in enumerate(ano_test_loaders):
            ano_preds = get_predictions(ano_loader, threshold)
            confusion_mat[real_class, real_class] += np.sum(real_preds == 1)
            confusion_mat[real_class, i if i < real_class else i + 1] += np.sum(ano_preds == 0)

        # Save the model weights after training for each class
        save_weights(vanilla_autoencoder, f'models/vanilla_autoencoder/vanilla_autoencoder_class_{real_class}_weights.pth')

    return confusion_mat, all_aurocs, all_thresholds, all_train_losses, all_val_losses

# Call the main function and get results
confusion_mat, all_aurocs, all_thresholds, all_train_losses, all_val_losses = main()

# Función para graficar las pérdidas
def plot_losses(train_losses, val_losses, class_index):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for Class {class_index}')
    plt.legend()
    plt.show()

# Función para graficar la matriz de confusión
def plot_confusion_matrix(confusion_mat):
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_mat, annot=True, fmt='g', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Función para graficar AUROC y umbrales
def plot_auroc_thresholds(aurocs, thresholds):
    plt.figure(figsize=(10, 5))
    plt.bar(range(10), aurocs, alpha=0.6, label='AUROC')
    plt.plot(range(10), thresholds, marker='o', linestyle='--', color='red', label='Threshold')
    plt.xlabel('Class')
    plt.ylabel('Value')
    plt.title('AUROC and Thresholds for Each Class')
    plt.legend()
    plt.show()

def load_weights(model, path):
    model.load_state_dict(torch.load(path)) # Load the saved model weights
    model.eval()  # Important: Set the model to evaluation mode

# Function to display 5 original and reconstructed images from the test dataset
def compare_images(test_loader):
    images, reconstructed_images = [], []
    for class_index in range(10):
        # Load the model for the specific class
        vanilla_autoencoder = VanillaAutoencoder(latent_dim).to(device)
        load_weights(vanilla_autoencoder, f'models/vanilla_autoencoder/vanilla_autoencoder_class_{class_index}_weights.pth')
        vanilla_autoencoder.eval()
        
        with torch.no_grad():
            for i, (inputs, _) in enumerate(test_loader):
                if i >= 5:
                    break
                inputs = inputs.to(device)
                outputs = vanilla_autoencoder(inputs)
                images.append(inputs.cpu().squeeze())
                reconstructed_images.append(outputs.cpu().squeeze())

    fig, axs = plt.subplots(5, 2, figsize=(10, 20))
    for i in range(5):
        axs[i, 0].imshow(images[i].permute(1, 2, 0).numpy())
        axs[i, 0].set_title("Original")
        axs[i, 0].axis("off")
        axs[i, 1].imshow(reconstructed_images[i].permute(1, 2, 0).numpy())
        axs[i, 1].set_title(f"Reconstructed (Class {i})")
        axs[i, 1].axis("off")
    plt.show()

# Display 5 original and reconstructed images from the test dataset using different models for each class
compare_images(test_loader)

# Function to add noise to images
def add_noise(img):
    noise = torch.randn_like(img) * 0.1
    img_noisy = torch.clamp(img + noise, 0., 1.)
    return img_noisy

# Function to train the denoising autoencoder for one epoch
def train_denoising_epoch(denoising_autoencoder, train_loader, criterion, optimizer):
    denoising_autoencoder.train()
    total_loss = 0
    for inputs, _ in train_loader:
        inputs = inputs.to(device)
        inputs_noisy = add_noise(inputs)
        optimizer.zero_grad()
        outputs = denoising_autoencoder(inputs_noisy)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Main function to train and evaluate the denoising autoencoder model
def main_denoising():
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize confusion matrix and lists to store results
    confusion_mat = np.zeros((10, 10))
    all_aurocs = []
    all_thresholds = []
    all_train_losses = []
    all_val_losses = []

    # Loop over each class as the real class
    for real_class in range(10):
        print(f'Training for real class: {real_class}')

        # Load datasets
        train_dataset = OneClassDatasetCIFAR10(root_dir='data', real_class=real_class, transform=transform, train=True, download=True)
        val_dataset = OneClassDatasetCIFAR10(root_dir='data', real_class=real_class, transform=transform, train=False, download=True)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

        real_test_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        ano_test_loaders = [DataLoader(OneClassDatasetCIFAR10(root_dir='data', real_class=i, transform=transform, train=False, download=True), batch_size=32, shuffle=False) for i in range(10) if i != real_class]

        # Initialize model, criterion, and optimizer
        denoising_autoencoder = VanillaAutoencoder(latent_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(denoising_autoencoder.parameters(), lr=0.0005)

        # Training loop
        num_epochs = 20
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            train_loss = train_denoising_epoch(denoising_autoencoder, train_loader, criterion, optimizer)
            val_loss = evaluate_model(denoising_autoencoder, val_loader, criterion)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Store losses
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        # Calculate AUROC
        auroc = test_model(denoising_autoencoder, real_test_loader, ano_test_loaders)
        all_aurocs.append(auroc)
        print(f'AUROC for class {real_class}: {auroc:.4f}')

        # Select threshold
        threshold = select_threshold(denoising_autoencoder, real_test_loader, ano_test_loaders[0])
        all_thresholds.append(threshold)
        print(f'Selected Threshold for class {real_class}: {threshold:.4f}')

        # Update confusion matrix
        def get_predictions(loader, threshold):
            predictions = []
            with torch.no_grad():
                for inputs, _ in loader:
                    inputs = inputs.to(device)
                    outputs = denoising_autoencoder(inputs)
                    mse_loss = F.mse_loss(outputs, inputs, reduction='none').mean(dim=[1, 2, 3])
                    preds = (mse_loss < threshold).cpu().numpy()
                    predictions.append(preds)
            return np.concatenate(predictions)

        real_preds = get_predictions(real_test_loader, threshold)
        for i, ano_loader in enumerate(ano_test_loaders):
            ano_preds = get_predictions(ano_loader, threshold)
            confusion_mat[real_class, real_class] += np.sum(real_preds == 1)
            confusion_mat[real_class, i if i < real_class else i + 1] += np.sum(ano_preds == 0)

        # Save the model weights after training for each class
        save_weights(denoising_autoencoder, f'models/denoising_autoencoder/denoising_autoencoder_class_{real_class}_weights.pth')

    return confusion_mat, all_aurocs, all_thresholds, all_train_losses, all_val_losses

# Call the main function for denoising autoencoder and get results
confusion_mat, all_aurocs, all_thresholds, all_train_losses, all_val_losses = main_denoising()