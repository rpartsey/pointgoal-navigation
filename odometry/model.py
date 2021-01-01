from abc import ABC, abstractmethod

import torch.nn as nn
import torchvision.models as models


class VONetBaseModel(nn.Module, ABC):
    def __init__(self, encoder, fc):
        super().__init__()
        self.encoder = encoder
        self.fc = fc

    @abstractmethod
    def forward(self, x):
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config):
        pass

    @staticmethod
    def create_fc_layers(input_size: int, hidden_size: list, output_size: int, p_dropout: float = 0.0):
        hidden_size.insert(0, input_size)

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
            nn.Linear(hidden_size[-1], output_size)
        )


class VONetResnet18(VONetBaseModel):
    def forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        x = self.fc(x)

        return x

    @classmethod
    def from_config(cls, config):
        resnet18 = models.resnet18()

        default_conv1 = resnet18.conv1
        resnet18.conv1 = nn.Conv2d(
            config.in_channels,
            default_conv1.out_channels,
            kernel_size=default_conv1.kernel_size,
            stride=default_conv1.stride,
            padding=default_conv1.padding,
            bias=default_conv1.bias
        )

        default_fc = resnet18.fc
        fc = cls.create_fc_layers(
            input_size=default_fc.in_features,
            hidden_size=config.hidden_size,
            output_size=config.output_dim,
            p_dropout=config.p_dropout
        )

        del resnet18.avgpool
        del resnet18.fc

        return cls(resnet18, fc)
