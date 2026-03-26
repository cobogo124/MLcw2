import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Ensure the Windows-specific fix for multi-processing is active
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 

from model import TypiClustResNet
from originalTypiClust import select_queries as select_original
from modifiedTypiClust import select_queries_modified as select_modified

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # 2. Load Pretrained Model
    model = TypiClustResNet(use_projection=True).to(device)
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'simclr_cifar10.pth')
    
    if os.path.exists(checkpoint_path):
        # Load backbone weights for feature extraction [cite: 275]
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
        print("Pretrained weights loaded.")
    else:
        print("[!] Warning: simclr_cifar10.pth not found. Results will be near random.")
    model.eval()

    # 3. Feature Extraction with L2 Normalization [cite: 956]
    def get_features(loader):
        feats, labs = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                # Extract features from the penultimate layer [cite: 275]
                f = model.backbone(imgs.to(device))
                f = torch.flatten(f, 1)
                # Apply L2 normalization to ensure stable distance metrics 
                f = F.normalize(f, p=2, dim=1)
                feats.append(f.cpu().numpy())
                labs.append(labels.numpy())
        return np.concatenate(feats), np.concatenate(labs)

    print("Extracting normalized semantic features...")
    X_train, y_train = get_features(train_loader)
    X_test, y_test = get_features(test_loader)

    # 4. Evaluation Loop
    # TypiClust is specifically designed for these extreme low-budget regimes [cite: 80, 255, 302]
    budgets = [10, 50, 100, 200] 
    acc_orig, acc_mod = [], []

    for b in budgets:
        print(f"Testing budget: {b} labels...")
        
        # Run original and modified selection logic
        idx_o = select_original(X_train, b)
        idx_m = select_modified(X_train, b)
        
        def train_eval(indices):
            # Evaluate using a linear classifier on fixed features [cite: 317]
            clf = LogisticRegression(max_iter=1000, solver='lbfgs')
            clf.fit(X_train[indices], y_train[indices])
            return accuracy_score(y_test, clf.predict(X_test))

        acc_orig.append(train_eval(idx_o))
        acc_mod.append(train_eval(idx_m))

    # 5. Output Results Table
    print("\n" + "="*45)
    print("      ACTIVE LEARNING PERFORMANCE GAINS")
    print("="*45)
    print(f"{'Budget':<8} | {'Original':<10} | {'Modified':<10} | {'Gain'}")
    print("-" * 45)
    for i, b in enumerate(budgets):
        gain = acc_mod[i] - acc_orig[i]
        print(f"{b:<8} | {acc_orig[i]:.4f}     | {acc_mod[i]:.4f}     | {gain:+.4f}")
    print("="*45)

    # 6. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(budgets, acc_orig, 'o--', color='gray', label='Original TypiClust')
    plt.plot(budgets, acc_mod, 's-', color='blue', label='Modified TypiClust (K=20)', linewidth=2)
    plt.xlabel('Number of Labeled Samples')
    plt.ylabel('Test Accuracy')
    plt.title('Low-Budget Performance: Original vs. K-tuning')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Dynamic filename to avoid Permission Denied errors
    import time
    timestamp = int(time.time())
    save_path = f'accuracy_comparison_{timestamp}.png'
    
    try:
        plt.savefig(save_path)
        print(f"\n[+] Plot successfully saved as: {save_path}")
    except PermissionError:
        print(f"\n[!] Error: Close the image viewer and delete {save_path} before re-running.")
    
    plt.show()

if __name__ == "__main__":
    main()