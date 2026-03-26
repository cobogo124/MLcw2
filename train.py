import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from model import TypiClustResNet

# Augmentation Strategy
simclr_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=32), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    def __getitem__(self, index):
        img, _ = self.dataset[index]
        return self.transform(img), self.transform(img)
    def __len__(self):
        return len(self.dataset)

def contrastive_loss(out_1, out_2, temperature=0.5):
    # Ensure all tensors are created on the same device as the input data
    device = out_1.device
    out = torch.cat([out_1, out_2], dim=0)
    n = out.shape[0]
    
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix, device=device) - torch.eye(n, device=device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(n, -1)
    
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    return (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting training on: {device}") # Immediate confirmation
    
    base_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    # Added num_workers=0 to prevent Windows multi-processing hangs
    train_loader = DataLoader(SimCLRDataset(base_dataset, simclr_transforms), 
                              batch_size=512, 
                              shuffle=True, 
                              num_workers=8,
                              pin_memory=True,
                              persistent_workers=True)

    model = TypiClustResNet(use_projection=True).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.4, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

    model.train()
    for epoch in range(500):
        running_loss = 0.0
        for batch_idx, (view_1, view_2) in enumerate(train_loader):
            view_1, view_2 = view_1.to(device), view_2.to(device)
            out_1, out_2 = model(view_1), model(view_2)
            
            out_1, out_2 = F.normalize(out_1, dim=1), F.normalize(out_2, dim=1)
            loss = contrastive_loss(out_1, out_2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # NEW: Print every 20 batches so you know it's working!
            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch}/500] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        scheduler.step()
        print(f"--- Epoch {epoch} Avg Loss: {running_loss/len(train_loader):.4f} ---")

    torch.save(model.state_dict(), 'simclr_cifar10.pth')

if __name__ == "__main__":
    train()