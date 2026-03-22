import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from model import TypiClustResNet

# Augmentation Strategy
# teach the model to ignore color/crops and focus on semantics
simclr_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=32), # CIFAR-10 size 
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Wrapper to return two views of the same image [cite: 566]
class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    def __getitem__(self, index):
        img, _ = self.dataset[index]
        return self.transform(img), self.transform(img)
    def __len__(self):
        return len(self.dataset)

# 2. NT-Xent Loss (Contrastive Loss)
# This math penalizes the model if the two views of the same image are far apart
def contrastive_loss(out_1, out_2, temperature=0.5):
    out = torch.cat([out_1, out_2], dim=0)
    n = out.shape[0]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(n, device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(n, -1)
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    return (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

# 3. Training Loop [cite: 953, 954]
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    train_loader = DataLoader(SimCLRDataset(base_dataset, simclr_transforms), batch_size=512, shuffle=True)

    model = TypiClustResNet(use_projection=True).to(device)
    # The paper uses SGD with 0.9 momentum and 0.0001 weight decay [cite: 953, 954]
    optimizer = optim.SGD(model.parameters(), lr=0.4, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

    model.train()
    for epoch in range(500): # 500 epochs as per paper 
        for (view_1, view_2) in train_loader:
            view_1, view_2 = view_1.to(device), view_2.to(device)
            out_1, out_2 = model(view_1), model(view_2)
            
            # Normalize for cosine similarity calculation
            out_1, out_2 = F.normalize(out_1, dim=1), F.normalize(out_2, dim=1)
            
            loss = contrastive_loss(out_1, out_2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch} complete.")

    torch.save(model.state_dict(), 'simclr_cifar10.pth')

if __name__ == "__main__":
    train()