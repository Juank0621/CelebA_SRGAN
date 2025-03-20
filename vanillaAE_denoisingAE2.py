import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10

# Define transformations for the CIFAR10 dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

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

data_dir= 'data/celeba'

# Custom Dataset class for loading images from a single folder
class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        # Return a dummy label because the autoencoder does not need labels
        return image, 0

# Load the CelebA dataset from the local folder
dataset = CelebADataset(root_dir=data_dir, transform=None)

# Verify the number of images in dataset
print(f'Number of images in dataset: {len(dataset)}')

# Function to show real images 
def show_real_images(image_paths, transform=None):
    images = []
    for img_path in image_paths:
        image = Image.open(img_path)
        if transform:
            image = transform(image)
        images.append(np.array(image))  # Convert to numpy array
    
    # Create subplots with a fixed aspect ratio and a layout proportional to the images
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    if len(images) == 1:  # If there's only one image, axes is not an array
        axes = [axes]
    
    for i, img in enumerate(images):
        if img.shape[0] == 3:  
            img = np.transpose(img, (1, 2, 0))  # Convert from CHW to HWC format
        # Show image
        axes[i].imshow(img)
        # Set the axis labels and ticks as proportional
        axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[i].yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Turn on axis
        axes[i].axis('on')
        
    plt.tight_layout()
    plt.show()

# Example usage of show_real_images
image_paths = [os.path.join(dataset.root_dir, dataset.image_files[i]) for i in range(5)]
show_real_images(image_paths)

# Here we define the transformations to be applied to the images in the CelebA dataset.
# We resize the images to 64x64 pixels and convert them to tensors.
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])


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

# Define the optimizer and loss function
criterion = nn.MSELoss()
optimizer = optim.Adam(vanilla_autoencoder.parameters(), lr=0.001)

# Initialize TensorBoard writer
writer = SummaryWriter("tb_logs/vanilla_autoencoder")

# Define transformations for the CIFAR10 dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load the CIFAR10 dataset
train_dataset = CIFAR10(root='data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='data', train=False, download=True, transform=transform)

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

# Show images from the CIFAR10 dataset
show_transform_images(train_loader)

# Training loop
num_epochs = 20
train_losses = []  # List to store training losses
val_losses = []  # List to store validation losses

for epoch in range(num_epochs):  
    vanilla_autoencoder.train()
    total_loss = 0  # Track total loss for monitoring

    # Use tqdm to add a progress bar to the training
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as tepoch:
        for img, _ in tepoch:
            img = img.to(device).float()  # Ensure images are float
            
            optimizer.zero_grad()
            output = vanilla_autoencoder(img)
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()  # Accumulate batch loss

            # Update progress bar with the loss value
            tepoch.set_postfix(loss=loss.item())

    # Average loss per epoch
    avg_loss = total_loss / len(train_loader)  # Compute average loss
    train_losses.append(avg_loss)  # Save the average loss
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}')

    # Log the average loss to TensorBoard
    writer.add_scalar("Loss/Train", avg_loss, epoch)

    # Validation loop
    vanilla_autoencoder.eval()
    val_loss = 0
    with torch.no_grad():
        for img, _ in val_loader:
            img = img.to(device).float()
            output = vanilla_autoencoder(img)
            loss = criterion(output, img)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.6f}')
    writer.add_scalar("Loss/Validation", avg_val_loss, epoch)

# Save the trained model after training
torch.save(vanilla_autoencoder.state_dict(), 'vanilla_autoencoder.pth')
print("Model saved.")

# Plot training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.show()

# Close the TensorBoard writer
writer.close()

# Load the trained model in a new session
vanilla_autoencoder = VanillaAutoencoder(latent_dim).to(device) # Initialize a new instance of the model
vanilla_autoencoder.load_state_dict(torch.load('vanilla_autoencoder.pth')) # Load the saved model weights
vanilla_autoencoder.eval()  # Important: Set the model to evaluation mode
print("Model loaded and ready to use.")

# Function to compare one real image with the image reconstructed by the decoder
def compare_real_and_reconstructed(loader, model, index=1):
    model.eval()
    with torch.no_grad():
        for img, _ in loader:
            img = img.to(device)
            output = model(img)
            img = img.cpu().numpy()
            output = output.cpu().numpy()
            
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            # Transponer las imágenes de (C, H, W) a (H, W, C)
            real_img = np.transpose(img[index], (1, 2, 0))
            reconstructed_img = np.transpose(output[index], (1, 2, 0))
            
            # Mostrar las imágenes
            axes[0].imshow(real_img)
            axes[0].set_title('Real Image')
            axes[0].set_xticks(np.arange(0, img.shape[2], 16))
            axes[0].set_yticks(np.arange(0, img.shape[3], 16))
            
            axes[1].imshow(reconstructed_img)
            axes[1].set_title('Reconstructed Image')
            axes[1].set_xticks(np.arange(0, output.shape[2], 16))
            axes[1].set_yticks(np.arange(0, output.shape[3], 16))
            
            plt.show()
            break

# Compare one real image with the reconstructed image
compare_real_and_reconstructed(test_loader, vanilla_autoencoder)

# Function to add noise to images
# This function adds Gaussian noise to the input images.
# The noise level is controlled by the standard deviation of the Gaussian distribution.
def add_noise(img):
    noise = torch.randn_like(img) * 0.1
    img_noisy = torch.clamp(img + noise, 0., 1.)
    return img_noisy

denoising_autoencoder = VanillaAutoencoder(latent_dim).to(device)
optimizer = optim.Adam(denoising_autoencoder.parameters(), lr=0.001)

# Training loop for denoising autoencoder
for epoch in range(num_epochs):
    denoising_autoencoder.train()
    running_loss = 0.0
    for batch in train_loader:
        img, _ = batch
        img = img.to(device)
        img_noisy = add_noise(img)
        
        optimizer.zero_grad()
        output = denoising_autoencoder(img_noisy)
        loss = criterion(output, img)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

# Save the trained denoising autoencoder model after training
torch.save(denoising_autoencoder.state_dict(), 'denoising_autoencoder.pth')
print("Denoising Autoencoder model saved.")

# Load the trained denoising autoencoder model for testing
denoising_autoencoder.load_state_dict(torch.load('denoising_autoencoder.pth'))
denoising_autoencoder.eval()
print("Denoising Autoencoder model loaded and ready to use.")

# Function to show original and reconstructed images
# This function visualizes the original and reconstructed images.
# It takes a data loader and the trained model as input, and displays a few examples of original and reconstructed images.
def show_images(loader, model):
    model.eval()
    with torch.no_grad():
        for img, _ in loader:
            img = img.to(device)
            output = model(img)
            img = img.cpu().numpy()
            output = output.cpu().numpy()
            fig, axes = plt.subplots(2, 5, figsize=(10, 4))
            for i in range(5):
                axes[0, i].imshow(np.transpose(img[i], (1, 2, 0)))  # Ensure values are in [0, 1]
                axes[0, i].set_xticks(np.arange(0, img.shape[2] + 1, img.shape[2]))
                axes[0, i].set_yticks(np.arange(0, img.shape[3] + 1, img.shape[3]))
                axes[0, i].axis('on')  # Keep axes visible
                axes[1, i].imshow(np.transpose(output[i], (1, 2, 0)))  # Ensure values are in [0, 1]
                axes[1, i].set_xticks(np.arange(0, output.shape[2] + 1, output.shape[2]))
                axes[1, i].set_yticks(np.arange(0, output.shape[3] + 1, output.shape[3]))
                axes[1, i].axis('on')  # Keep axes visible
            plt.show()
            break

show_images(test_loader, denoising_autoencoder)
summary(vanilla_autoencoder, input_size=(3, 128, 128))