import torch
import torch.nn as nn
from segmentation_models_pytorch.encoders import get_encoder


class VONet(nn.Module):
    def __init__(self, encoder, fc):
        super().__init__()
        self.encoder = encoder
        self.fc = fc

    def forward(self, x):
        x = self.encoder(x)[-1]  # get last stage output
        x = self.fc(x)

        return x

    @classmethod
    def from_config(cls, model_config):
        model_params = model_config.params
        encoder_params = model_params.encoder.params
        fc_params = model_params.fc.params

        encoder = get_encoder(
            name=model_params.encoder.type,
            in_channels=encoder_params.in_channels,
            depth=encoder_params.depth,
            weights=encoder_params.weights
        )

        fc = cls.create_fc_layers(
            input_size=cls.compute_output_size(encoder, encoder_params),
            hidden_size=fc_params.hidden_size,
            output_size=fc_params.output_size,
            p_dropout=fc_params.p_dropout
        )

        return cls(encoder, fc)

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

    @staticmethod
    def compute_output_size(encoder, config):
        input_size = (1, config.in_channels, config.in_height, config.in_width)

        encoder_input = torch.randn(*input_size)
        with torch.no_grad():
            output = encoder(encoder_input)

        return output[-1].view(-1).size(0)
