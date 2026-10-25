import torch.nn as nn


class SmallCNN(nn.Module):
    """Three pooled convolutions; logits and a shared penultimate feature API."""
    def __init__(self, width=16, feature_dim=64):
        super().__init__()
        blocks = []
        incoming = 3
        for outgoing in (width, width * 2, width * 4):
            blocks.extend([nn.Conv2d(incoming, outgoing, 3, padding=1),
                           nn.ReLU(), nn.MaxPool2d(2)])
            incoming = outgoing
        self.encoder = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten())
        self.features = nn.Sequential(nn.Linear(incoming * 4, feature_dim), nn.ReLU())
        self.head = nn.Linear(feature_dim, 10)

    def forward(self, x):
        features = self.features(self.encoder(x))
        return self.head(features), features


def build_model(cfg):
    architecture = cfg.get('architecture', 'small_cnn')
    if architecture == 'small_cnn':
        return SmallCNN(cfg['width'], cfg['feature_dim'])
    if architecture == 'resnet18':
        return CIFARResNet18()
    raise ValueError(f'Unknown architecture: {architecture}')


class CIFARResNet18(nn.Module):
    """ResNet-18 from scratch with a CIFAR 3x3/stride-1 stem, no max pool."""
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18
        network = resnet18(weights=None, num_classes=10)
        network.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        network.maxpool = nn.Identity()
        self.encoder = nn.Sequential(*list(network.children())[:-1], nn.Flatten())
        self.head = network.fc

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features), features
