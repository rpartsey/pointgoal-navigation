import torch.nn as nn
import torchvision.models as models


class VONet(nn.Module):
    def __init__(self, encoder, fc):
        super().__init__()
        self.encoder = encoder
        self.fc = fc

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)

        return x

    @staticmethod
    def create_fc_layers(in_features, p_dropout=0.0):
        return nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Dropout(p_dropout),
            nn.ReLU(True),
            nn.Linear(512, 4)
        )

    @classmethod
    def from_torchvision_resnet18(cls):
        resnet18 = models.resnet18()
        fc = cls.create_fc_layers(resnet18.fc.in_features)
        del resnet18.fc

        return cls(resnet18, fc)
