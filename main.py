import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import TypiClustResNet
from typiClust import select_queries

def main():
    # Detect if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load CIFAR-10 Unlabeled Pool
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    # 2. Initialize Model and move to Device
    model = TypiClustResNet(use_projection=False) 
    model.to(device) # Move model weights to GPU
    
    try:
        state_dict = torch.load('simclr_cifar10.pth', map_location=device)
    except FileNotFoundError:
        print("Warning: simclr_cifar10.pth not found. Using random weights.")
        
    model.eval()

    # 3. Extract Features for the entire pool
    print("Extracting semantic features on GPU...")
    all_features = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device) # Move data batch to GPU
            feat = model(imgs) 
            # Move results back to CPU for numpy compatibility
    budget_b = 10 
    selected_indices = select_queries(features_matrix, budget_b)

    print(f"Top {budget_b} indices to label first:")
    print(selected_indices)

if __name__ == "__main__":
    main()