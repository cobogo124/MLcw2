import torch
import torch.nn as nn
import torchvision.models as models

class TypiClustResNet(nn.Module):
    def __init__(self, use_projection=True):
        super(TypiClustResNet, self).__init__()
        
        # FIX: Changed 'pretrained=False' to 'weights=None' to remove warning
        resnet = models.resnet18(weights=None)
        
        # Strip the final fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        self.use_projection = use_projection
        
        if self.use_projection:
            self.projection_head = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 128) 
            )

    def forward(self, x):
        features = torch.flatten(self.backbone(x), 1)
        if self.use_projection:
            return self.projection_head(features)
        return features