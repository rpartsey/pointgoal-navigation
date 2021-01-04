from abc import ABC, abstractmethod

import torch
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
        model_params = config.params

        resnet18 = models.resnet18()

        default_conv1 = resnet18.conv1
        resnet18.conv1 = nn.Conv2d(
            model_params.in_channels,
            default_conv1.out_channels,
            kernel_size=default_conv1.kernel_size,
            stride=default_conv1.stride,
            padding=default_conv1.padding,
            bias=default_conv1.bias
        )

        input_size = cls.compute_output_size(resnet18, model_params)
        fc = cls.create_fc_layers(
            input_size=input_size,
            hidden_size=model_params.hidden_size,
            output_size=model_params.output_dim,
            p_dropout=model_params.p_dropout
        )

        del resnet18.avgpool
        del resnet18.fc

        return cls(resnet18, fc)

    @staticmethod
    def compute_output_size(encoder, config):
        input_size = (1, config.in_channels, config.in_height, config.in_width)

        encoder_input = torch.randn(*input_size)
        with torch.no_grad():
            x = encoder_input

            x = encoder.conv1(x)
            x = encoder.bn1(x)
            x = encoder.relu(x)
            x = encoder.maxpool(x)

            x = encoder.layer1(x)
            x = encoder.layer2(x)
            x = encoder.layer3(x)
            x = encoder.layer4(x)

            output = x

        return output.view(-1).size(0)
