import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import lightning as L
from sklearn.metrics import roc_curve, auc

data_dir = 'data/celeba'

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
        label = np.random.randint(0, 4)
        image = transforms.functional.rotate(image, label * 90)
        return image, label

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = CelebADataset(root_dir=data_dir, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(8 * 8 * 256, latent_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8 * 8 * 256),
            nn.ReLU(),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)

class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

latent_dim = 256
autoencoder = Autoencoder(latent_dim)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autoencoder.to(device)

class AutoencoderLightningModule(L.LightningModule):
    def __init__(self, latent_dim):
        super(AutoencoderLightningModule, self).__init__()
        self.autoencoder = Autoencoder(latent_dim)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.autoencoder(x)

    def training_step(self, batch, batch_idx):
        img, _ = batch
        output = self.autoencoder(img)
        loss = self.criterion(output, img)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.autoencoder.parameters(), lr=0.001)

model = AutoencoderLightningModule(latent_dim)

trainer = L.Trainer(max_epochs=10, devices=device)
trainer.fit(model, train_loader)

trainer.save_checkpoint('autoencoder_anomaly.ckpt')
print("Model saved.")

model = AutoencoderLightningModule.load_from_checkpoint('autoencoder_anomaly.ckpt')
model.eval()
print("Model loaded and ready to use.")

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
                axes[0, i].imshow(np.transpose(img[i], (1, 2, 0)))
                axes[0, i].axis('on')
                axes[1, i].imshow(np.transpose(output[i], (1, 2, 0)))
                axes[1, i].axis('on')
            plt.show()
            break

show_images(test_loader, model)

def calculate_anomaly_score(model, loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for img, _ in loader:
            img = img.to(device)
            output = model(img)
            loss = nn.functional.mse_loss(output, img, reduction='none')
            loss = loss.view(loss.size(0), -1).mean(dim=1)
            scores.append(loss.cpu().numpy())
    return np.concatenate(scores)

def evaluate_anomaly_detection(model, real_loader, anomaly_loader):
    real_scores = calculate_anomaly_score(model, real_loader)
    anomaly_scores = calculate_anomaly_score(model, anomaly_loader)
    y_true = np.concatenate([np.zeros_like(real_scores), np.ones_like(anomaly_scores)])
    y_scores = np.concatenate([real_scores, anomaly_scores])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    return roc_auc

real_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
anomaly_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

roc_auc = evaluate_anomaly_detection(model, real_loader, anomaly_loader)
print(f'ROC AUC for anomaly detection: {roc_auc}')
