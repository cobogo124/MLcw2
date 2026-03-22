import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import TypiClustResNet
from typiClust import select_queries

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load CIFAR-10 Unlabeled Pool
    # We use basic transforms (no heavy augmentations) for feature extraction
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    model = TypiClustResNet(use_projection=False) # We want features, not projection 
    state_dict = torch.load('simclr_cifar10.pth', map_location=device)
    model.load_state_dict(state_dict, strict=False) # strict=False ignores the projection head weights 
    model.to(device)
    model.eval()

    # 3. Extract Features for the entire pool
    print("Extracting semantic features...")
    all_features = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            feat = model(imgs) # Output is the penultimate layer
            all_features.append(feat.cpu().numpy())
    
    features_matrix = np.concatenate(all_features, axis=0)

    # 4. Run TypiClust Selection
    # For CIFAR-10, B=10 is a common low-budget starting point
    budget_b = 10 
    selected_indices = select_queries(features_matrix, budget_b)

    print(f"Top {budget_b} indices to label first:")
    print(selected_indices)
    
    # Optional: Visualize or save the selected image filenames
    # In a real workflow, you would now send these to an expert for tagging

if __name__ == "__main__":
    main()