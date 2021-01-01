from abc import ABC, abstractmethod

import torch.nn as nn
import torchvision.models as models


class VONetBaseModel(nn.Module, ABC):
    def __init__(self, encoder, fc):
        super().__init__()
        self.encoder = encoder
        self.fc = fc

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)

        return x

    @staticmethod
    def create_fc_layers(input_dim: int, hidden_size: list, output_dim: int, p_dropout: float = 0.0):
        hidden_size.insert(0, input_dim)

        layers = []
        for i in range(len(hidden_size)-1):
            layers += [
                nn.Dropout(p_dropout),
                nn.Linear(hidden_size[i], hidden_size[i+1]),
                nn.ReLU(True),
            ]

        return nn.Sequential(
            nn.Flatten(),
            *layers,
            nn.Dropout(p_dropout),
            nn.Linear(hidden_size[-1], output_dim)
        )

    @classmethod
    @abstractmethod
    def from_config(cls, config):
        pass


class VONetResnet18(VONetBaseModel):
    @classmethod
    def from_config(cls, config):
        resnet18 = models.resnet18()
        fc = cls.create_fc_layers(
            input_dim=config.input_dim,
            hidden_size=config.hidden_size,
            output_dim=config.output_dim,
            p_dropout=config.p_dropout
        )
        del resnet18.fc

        return cls(resnet18, fc)
