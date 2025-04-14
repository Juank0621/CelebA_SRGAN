# Import necessary libraries
import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm.auto import tqdm
from math import log10

from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision('medium')

# Set matrix multiplication precision to medium for better performance
torch.set_float32_matmul_precision('medium')

# Display PyTorch version, CUDA version, and cuDNN version
print("PyTorch Version:", torch.__version__)

cuda_version = torch.version.cuda
print("CUDA Version:", cuda_version)

cudnn_version = torch.backends.cudnn.version()
print("cuDNN Version:", cudnn_version)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Num GPUs Available:", torch.cuda.device_count())

# Define transformations for high-resolution (HR) and low-resolution (LR) images
transform_hr = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to 128x128
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

transform_lr = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to 64x64
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Define the CelebA dataset class
class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform_hr=None, transform_lr=None):
        """
        Custom dataset class for loading CelebA images.

        Args:
            image_paths (list): List of image file paths.
            transform_hr (callable, optional): Transformation for high-resolution images.
            transform_lr (callable, optional): Transformation for low-resolution images.
        """
        self.image_paths = image_paths  # List of image file paths
        self.transform_hr = transform_hr  # Transformation for high-resolution images
        self.transform_lr = transform_lr  # Transformation for low-resolution images

    def __getitem__(self, idx):
        """
        Retrieve a high-resolution and low-resolution image pair.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: High-resolution and low-resolution images as tensors.
        """
        # Open the image at the given index
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')  # Open image and convert to RGB
        
        # Apply transformations for HR and LR images (convert to tensor)
        hr_img = self.transform_hr(img) if self.transform_hr else transforms.ToTensor()(img)
        lr_img = self.transform_lr(img) if self.transform_lr else transforms.ToTensor()(img)
        
        return hr_img, lr_img  # Return HR and LR images as tensors

    def __len__(self):
        """
        Get the total number of images in the dataset.

        Returns:
            int: Total number of images.
        """
        return len(self.image_paths)

# Split dataset into train and test sets
def split_dataset(root, test_ratio=0.2):
    """
    Split the dataset into training and testing sets.

    Args:
        root (str): Path to the dataset directory containing images.
        test_ratio (float): Proportion of the dataset to use for testing.

    Returns:
        tuple: Paths for training and testing datasets.
    """
    image_paths = [os.path.join(root, img) for img in os.listdir(root) if img.endswith('.jpg')]
    split_idx = int(len(image_paths) * (1 - test_ratio))
    train_paths = image_paths[:split_idx]
    test_paths = image_paths[split_idx:]
    return train_paths, test_paths

# Paths to the dataset
dataset_root = './data/celeba'
train_paths, test_paths = split_dataset(dataset_root)

# Create train and test datasets
train_dataset = CelebADataset(train_paths, transform_hr=transform_hr, transform_lr=transform_lr)
test_dataset = CelebADataset(test_paths, transform_hr=transform_hr, transform_lr=transform_lr)

# Display the number of images in the training and testing datasets
print(f"Number of images in the training dataset: {len(train_dataset)}")
print(f"Number of images in the testing dataset: {len(test_dataset)}")

# Create DataLoaders for train and test datasets
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

def show_images(images_real, images_transformed, ncols=5, nrows=2):
    def denormalize(img):
        """
        Denormalizes an image from the range [-1, 1] to the range [0, 1].
        """
        return (img * 0.5) + 0.5  # Reverse normalization

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(ncols * nrows):
        if i < 10:
            img = denormalize(images_real[i]).numpy().transpose(1, 2, 0)  # Denormalize and convert to HWC
        else:
            img = denormalize(images_transformed[i - 10]).numpy().transpose(1, 2, 0)  # Denormalize and convert to HWC
        
        axes[i].imshow(img)
        axes[i].axis('off')

        # Adjust axis ticks
        axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))  # Integer ticks on x-axis
        axes[i].yaxis.set_major_locator(MaxNLocator(integer=True))  # Integer ticks on y-axis
    
    plt.tight_layout()
    plt.show()

# Define the Residual Block
class ResidualBlock(nn.Module):
    """
    Residual Block used in the Generator model.

    Args:
        channels (int): Number of input and output channels for the block.
    """
    def __init__(self, channels):
        """
        Initialize the Residual Block.

        Args:
            channels: The number of input and output channels for the block.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(),  
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        """
        Forward pass for the Residual Block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with residual connection.
        """
        # Return the input with the residual connection added to the output of the block
        return x + self.layers(x)

# Define the Generator (SRResNet)
class Generator(nn.Module):
    """
    Generator model for Super-Resolution (SRResNet).

    Args:
        base_channels (int): Number of channels in the first convolutional layer.
        n_ps_blocks (int): Number of PixelShuffle blocks.
        n_res_blocks (int): Number of Residual blocks.
    """
    def __init__(self, base_channels=64, n_ps_blocks=2, n_res_blocks=16):
        """
        Initialize the Generator (SRResNet) model.

        Args:
            base_channels: The number of channels in the first convolutional layer.
            n_ps_blocks: The number of PixelShuffle blocks.
            n_res_blocks: The number of Residual blocks.
        """
        super().__init__()
        self.in_layer = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=9, padding=4),
            nn.SiLU(),  
        )
        res_blocks = [ResidualBlock(base_channels) for _ in range(n_res_blocks)]
        res_blocks += [
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
        ]
        self.res_blocks = nn.Sequential(*res_blocks)
        ps_blocks = []
        for _ in range(n_ps_blocks):
            ps_blocks += [
                nn.Conv2d(base_channels, 4 * base_channels, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.SiLU(),  
            ]
        self.ps_blocks = nn.Sequential(*ps_blocks)
        self.out_layer = nn.Sequential(
            nn.Conv2d(base_channels, 3, kernel_size=9, padding=4),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        Forward pass for the Generator model.

        Args:
            x (torch.Tensor): Low-resolution input tensor.

        Returns:
            torch.Tensor: High-resolution output tensor.
        """
        x_res = self.in_layer(x)
        x = x_res + self.res_blocks(x_res)
        x = self.ps_blocks(x)
        x = self.out_layer(x)
        return x

# Define the Discriminator
class Discriminator(nn.Module):
    """
    Discriminator model for distinguishing real and fake high-resolution images.

    Args:
        base_channels (int): Number of channels in the first convolutional layer.
        n_blocks (int): Number of convolutional blocks.
    """
    def __init__(self, base_channels=64, n_blocks=3):
        super().__init__()
        self.blocks = [
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        cur_channels = base_channels
        for _ in range(n_blocks):
            self.blocks += [
                nn.Conv2d(cur_channels, 2 * cur_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(2 * cur_channels),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(2 * cur_channels, 2 * cur_channels, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(2 * cur_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            cur_channels *= 2
        self.blocks += [
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cur_channels, 2 * cur_channels, kernel_size=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * cur_channels, 1, kernel_size=1, padding=0),
            nn.Flatten(),
        ]
        self.layers = nn.Sequential(*self.blocks)

    def forward(self, x):
        """
        Forward pass for the Discriminator model.

        Args:
            x (torch.Tensor): Input tensor (image).

        Returns:
            torch.Tensor: Output tensor (real or fake prediction).
        """
        return self.layers(x)

# Define the Loss class
class Loss(nn.Module):
    """
    Loss function for training the SRGAN model.

    Args:
        device (str): Device to run the loss calculations ('cuda' or 'cpu').
    """
    def __init__(self, device='cuda'):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).to(device)
        self.vgg = nn.Sequential(*list(vgg.features)[:-1]).eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

    @staticmethod
    def img_loss(x_real, x_fake):
        """
        Calculate image reconstruction loss (MSE).

        Args:
            x_real (torch.Tensor): Real high-resolution image.
            x_fake (torch.Tensor): Generated high-resolution image.

        Returns:
            torch.Tensor: MSE loss.
        """
        return F.mse_loss(x_real, x_fake)

    def adv_loss(self, x, is_real):
        """
        Calculate adversarial loss.

        Args:
            x (torch.Tensor): Discriminator predictions.
            is_real (bool): Whether the predictions are for real images.

        Returns:
            torch.Tensor: Adversarial loss.
        """
        target = torch.zeros_like(x) if is_real else torch.ones_like(x)
        return F.binary_cross_entropy_with_logits(x, target)

    def vgg_loss(self, x_real, x_fake):
        """
        Calculate perceptual loss using VGG19 features.

        Args:
            x_real (torch.Tensor): Real high-resolution image.
            x_fake (torch.Tensor): Generated high-resolution image.

        Returns:
            torch.Tensor: Perceptual loss.
        """
        return F.mse_loss(self.vgg(x_real), self.vgg(x_fake))

    def forward(self, generator, discriminator, hr_real, lr_real):
        """
        Calculate generator and discriminator losses.

        Args:
            generator (nn.Module): Generator model.
            discriminator (nn.Module): Discriminator model.
            hr_real (torch.Tensor): Real high-resolution images.
            lr_real (torch.Tensor): Low-resolution images.

        Returns:
            tuple: Generator loss, discriminator loss, and generated high-resolution images.
        """
        hr_fake = generator(lr_real)
        fake_preds_for_g = discriminator(hr_fake)
        fake_preds_for_d = discriminator(hr_fake.detach())
        real_preds_for_d = discriminator(hr_real.detach())
        g_loss = (
            0.001 * self.adv_loss(fake_preds_for_g, False) +
            0.006 * self.vgg_loss(hr_real, hr_fake) +
            self.img_loss(hr_real, hr_fake)
        )
        d_loss = 0.5 * (
            self.adv_loss(real_preds_for_d, True) +
            self.adv_loss(fake_preds_for_d, False)
        )
        return g_loss, d_loss, hr_fake

# Function to calculate PSNR (Peak Signal-to-Noise Ratio)
def calculate_psnr(img1, img2, max_value=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        img1 (torch.Tensor): First image tensor.
        img2 (torch.Tensor): Second image tensor.
        max_value (float): Maximum pixel value (default: 1.0).

    Returns:
        float: PSNR value in decibels (dB).
    """
    # Convert tensor images to numpy arrays and calculate MSE
    img1 = img1.detach().cpu().numpy()  # Detach from computation graph and convert to numpy
    img2 = img2.detach().cpu().numpy()  # Detach from computation graph and convert to numpy
    mse = np.mean((img1 - img2) ** 2)
    
    # Avoid division by zero
    if mse == 0:
        return 100  # If MSE is 0, return a PSNR of 100
    psnr = 10 * log10((max_value ** 2) / mse)  # Calculate PSNR
    return psnr

# Function to calculate SSIM (Structural Similarity Index)
def calculate_ssim(img1, img2, win_size=3, data_range=2):
    """
    Calculate Structural Similarity Index (SSIM) between two images.

    Args:
        img1 (torch.Tensor): First image tensor.
        img2 (torch.Tensor): Second image tensor.
        win_size (int): Window size for SSIM calculation (default: 3).
        data_range (float): Data range of the images (default: 2).

    Returns:
        float: SSIM value.
    """
    # Convert tensor images to numpy arrays
    img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format for SSIM
    img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)
    
    # Compute SSIM with a smaller window size (e.g., 3 or 5)
    return ssim(img1, img2, win_size=win_size, multichannel=True, data_range=data_range)

# Define the training function
def train(generator, discriminator, dataloader, device, lr=1e-4, total_steps=1e4, display_step=1000, patience=5, min_improvement=0.1, warmup_steps=5000):
    """
    Train the Generator and Discriminator models using the given dataloader.

    Args:
        generator (nn.Module): The generator model.
        discriminator (nn.Module): The discriminator model.
        dataloader (DataLoader): DataLoader for the training dataset.
        device (str): Device to run the computations ('cuda' or 'cpu').
        lr (float, optional): Learning rate for the optimizers. Default is 1e-4.
        total_steps (int, optional): Total number of training steps. Default is 1e4.
        display_step (int, optional): Step interval for displaying metrics. Default is 1000.
        patience (int, optional): Number of steps without improvement in PSNR to trigger early stopping. Default is 5.
        min_improvement (float, optional): Minimum PSNR improvement required to reset early stopping. Default is 0.1.
        warmup_steps (int, optional): Number of steps to warm up before early stopping is considered. Default is 5000.

    Returns:
        None
    """

    generator = generator.to(device).train()
    discriminator = discriminator.to(device).train()
    loss_fn = Loss(device=device)

    g_optimizer = torch.optim.AdamW(generator.parameters(), lr=lr)
    d_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=lr)

    cur_step = 0
    mean_g_loss = 0.0
    mean_d_loss = 0.0
    best_psnr = -float('inf')
    patience_counter = 0

    pbar = tqdm(total=int(total_steps), desc="Training Steps")

    while cur_step < total_steps:
        for hr_real, lr_real in dataloader:
            if cur_step >= total_steps:
                break

            hr_real = hr_real.to(device)
            lr_real = lr_real.to(device)

            # Forward + backward
            g_loss, d_loss, hr_fake = loss_fn(generator, discriminator, hr_real, lr_real)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            mean_g_loss += g_loss.item() / display_step
            mean_d_loss += d_loss.item() / display_step

            # Métricas de validación
            psnr_value = calculate_psnr(hr_real[0], hr_fake[0])
            ssim_value = calculate_ssim(hr_real[0], hr_fake[0], win_size=3, data_range=2)

            # Mostrar durante warmup también
            if cur_step % display_step == 0:
                tqdm.write(f'Step {cur_step}: G_loss: {mean_g_loss:.5f}, D_loss: {mean_d_loss:.5f}')
                tqdm.write(f'PSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.4f}')
                mean_g_loss = 0.0
                mean_d_loss = 0.0

            # Early stopping
            if cur_step >= warmup_steps:
                if psnr_value > best_psnr + min_improvement:
                    best_psnr = psnr_value
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    tqdm.write(f"🛑 Early stopping at step {cur_step} due to no improvement in PSNR.")
                    return

            cur_step += 1
            pbar.update(1)

    pbar.close()

# Initialize and train the SRGAN
generator = Generator(n_res_blocks=16, n_ps_blocks=2).to(device)
discriminator = Discriminator(n_blocks=1, base_channels=8).to(device)

train(
    generator,  # Your generator model
    discriminator,  # Your discriminator model
    train_dataloader,  # Your dataloader for the training dataset
    device,  # The device ('cuda' or 'cpu')
    lr=1e-4,  # Learning rate
    total_steps=1e4,  # Total steps to train
    display_step=1000,  # Display metrics every 1000 steps
    patience=5,  # Early stopping patience (number of steps without improvement in PSNR)
    min_improvement=0.1,  # Minimum PSNR improvement required to reset early stopping
    warmup_steps=5000  # Number of steps to warm up before early stopping is considered
)

# Save the trained models
torch.save(generator.state_dict(), 'models/srgan/srgenerator.pth')
torch.save(discriminator.state_dict(), 'models/srgan/srdiscriminator.pth')


generator = Generator().to(device)
generator.load_state_dict(torch.load('models/srgan/srgenerator.pth', map_location=device))
generator.eval()

def evaluate(generator, test_loader, device):
    """
    Evaluate the generator model on the test dataset.

    Args:
        generator (nn.Module): The generator model.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (str): Device to run the computations ('cuda' or 'cpu').

    Returns:
        tuple: Average PSNR and SSIM values across the test dataset.
    """
    generator.eval()
    avg_psnr = 0.0
    avg_ssim = 0.0
    with torch.no_grad():
        for hr_real, lr_real in tqdm(test_loader, desc="Evaluating"):
            hr_real = hr_real.to(device)
            lr_real = lr_real.to(device)
            hr_fake = generator(lr_real)

            # Calculate PSNR and SSIM for the first image in the batch
            psnr_value = calculate_psnr(hr_real[0], hr_fake[0])
            ssim_value = calculate_ssim(hr_real[0], hr_fake[0], win_size=3, data_range=2)

            avg_psnr += psnr_value
            avg_ssim += ssim_value

    total = len(test_loader)
    avg_psnr /= total
    avg_ssim /= total
    print(f"\nEvaluation Results — PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim

def compare_images(generator, test_loader):
    """
    Compare low-resolution, super-resolution, and high-resolution images.

    Args:
        generator (nn.Module): The generator model.
        test_loader (DataLoader): DataLoader for the test dataset.

    Displays:
        A plot comparing 5 low-resolution, super-resolution, and high-resolution images.
    """

    def denormalize(tensor):
        # Convert image from [-1, 1] to [0, 1]
        return tensor * 0.5 + 0.5

    generator.eval()
    with torch.no_grad():
        for hr_real, lr_real in test_loader:
            lr_real = lr_real.to(device)
            sr_fake = generator(lr_real)

            # Display 5 low-resolution, super-resolution, and high-resolution images
            fig, axs = plt.subplots(3, 5, figsize=(15, 9))
            for i in range(5):
                lr_img = denormalize(lr_real[i].cpu()).permute(1, 2, 0).numpy()
                sr_img = denormalize(sr_fake[i].cpu()).permute(1, 2, 0).numpy()
                hr_img = denormalize(hr_real[i].cpu()).permute(1, 2, 0).numpy()

                axs[0, i].imshow(np.clip(lr_img, 0, 1))
                axs[0, i].set_title("Low Resolution")
                axs[0, i].axis("off")

                axs[1, i].imshow(np.clip(sr_img, 0, 1))
                axs[1, i].set_title("Super Resolution")
                axs[1, i].axis("off")

                axs[2, i].imshow(np.clip(hr_img, 0, 1))
                axs[2, i].set_title("High Resolution")
                axs[2, i].axis("off")

            plt.tight_layout()
            plt.show()
            break