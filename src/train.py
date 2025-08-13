import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import RealFakeCNN
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_model():
    # Force CPU usage
    device = torch.device('cpu')
    print(f"🖥️  Using device: CPU (No GPU detected)\n")

    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    # Dataset paths
    train_dir = os.path.join('data', 'train')
    test_dir = os.path.join('data', 'test')

    # Load datasets
    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    # Optionally reduce dataset size for quicker CPU testing:
    # from torch.utils.data import Subset
    # train_data = Subset(train_data, range(1000))
    # test_data = Subset(test_data, range(200))

    # DataLoaders
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # Model, Loss, Optimizer
    model = RealFakeCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("🟢 Starting training...\n")

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"📘 Epoch [{epoch+1}/{num_epochs}] completed. Total Loss: {total_loss:.4f}\n")

    # Save model
    os.makedirs("model", exist_ok=True)
    model_path = os.path.join("model", "real_fake_model.pth")
    torch.save(model.state_dict(), model_path)

    print(f"✅ Training complete. Model saved to: {model_path}")

if __name__ == "__main__":
    train_model()
