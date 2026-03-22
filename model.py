import torch
import torch.nn as nn
import torchvision.models as models

class TypiClustResNet(nn.Module):
    def __init__(self, use_projection=True):
        super(TypiClustResNet, self).__init__()
        
        # Load ResNet18 as the backbone for CIFAR-10
        resnet = models.resnet18(pretrained=False)
        
        # Strip the final fully connected layer to get the penultimate features
        # ResNet-18 penultimate layer size is 512
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        self.use_projection = use_projection
        
        # Manual implementation of the SimCLR Projection Head
        # This is what replaced the 'lightly' module
        if self.use_projection:
            self.projection_head = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 128) # Final projection to 128 dimensions
            )

    def forward(self, x):
        # 1. Extract features: (batch, 512, 1, 1) -> (batch, 512)
        features = torch.flatten(self.backbone(x), 1)
        
        # 2. If training SSL, use the projection head
        if self.use_projection:
            return self.projection_head(features)
        
        # 3. For TypiClust math, return the raw 512D penultimate features
        return features