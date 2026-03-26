import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import TypiClustResNet
from originalTypiClust import select_queries
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
            imgs = imgs.to(device)
            feat = model(imgs) 
            all_features.append(feat.cpu().numpy()) # Move to CPU and collect

    # Convert list to a single numpy matrix
    features_matrix = np.concatenate(all_features, axis=0)

    budget_b = 10 
    selected_indices = select_queries(features_matrix, budget_b)

    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features_matrix)

    # Plot all points
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c='gray', alpha=0.1, label='Pool')
    # Highlight selected points
    plt.scatter(reduced_features[selected_indices, 0], reduced_features[selected_indices, 1], c='red', label='Selected')
    plt.legend()
    plt.title("Active Learning Query Selection")
    plt.show()

if __name__ == "__main__":
    main()