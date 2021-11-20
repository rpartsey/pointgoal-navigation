import torch
import torch.nn as nn
from segmentation_models_pytorch.encoders import get_encoder
from habitat_baselines.rl.ddppo.policy import resnet


def init_distributed(model, device, find_unused_params: bool = True):
    if torch.cuda.is_available():
        ddp = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device],
            output_device=device,
            find_unused_parameters=find_unused_params,
        )
    else:
        ddp = nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=find_unused_params,
        )

    return ddp


class Normalize(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        if in_channels == 8:
            mean = torch.tensor(
                [
                    [[0.1749]],
                    [[0.5565]],
                    [[0.5044]],
                    [[0.4595]],
                    [[0.1735]],
                    [[0.5561]],
                    [[0.5043]],
                    [[0.4597]],
                ]
            )

            var = torch.tensor(
                [
                    [[0.0213]],
                    [[0.0442]],
                    [[0.0516]],
                    [[0.0617]],
                    [[0.0212]],
                    [[0.0444]],
                    [[0.0518]],
                    [[0.0619]],
                ]
            )
        else:
            raise RuntimeError(f"Don't know how to handle {in_channels} channels")

        self.register_buffer("_inv_stddev", torch.rsqrt(var.clamp(min=1e-2)))
        self.register_buffer("_negative_mean_mul_inv_sttdev", -mean * self._inv_stddev)

    def forward(self, x):
        return torch.addcmul(self._negative_mean_mul_inv_sttdev, self._inv_stddev, x)

class DropoutPart(nn.Module):
    def __init__(self, p, embedding_size):
        super().__init__()
        self.dropout = nn.Dropout(p, inplace=True)
        self.embedding_size = embedding_size

    def forward(self, x):
        self.dropout(x[:, self.embedding_size:])
        return x


class VONet(nn.Module):
    def __init__(self, encoder, fc, action_embedding=None, collision_embedding=None):
        super().__init__()
        self.action_embedding = action_embedding
        self.collision_embedding = collision_embedding
        self.encoder = encoder
        self.flatten = nn.Flatten()
        self.fc = fc

    def forward(self, x, action=None, collision=None):
        x = self.encoder(x)[-1]  # get last stage output
        x = self.flatten(x)
        if self.action_embedding:
            x = torch.cat([self.action_embedding(action).detach(), x], 1)
        if self.collision_embedding:
            x = torch.cat([self.collision_embedding(collision).detach(), x], 1)
        x = self.fc(x)

        return x

    @classmethod
    def from_config(cls, model_config):
        model_params = model_config.params
        encoder_params = model_params.encoder.params
        fc_params = model_params.fc.params

        action_embedding = nn.Embedding(
            num_embeddings=model_params.n_action_values,
            embedding_dim=model_params.action_embedding_size
        ) if model_params.action_embedding_size > 0 else None

        collision_embedding = nn.Embedding(
            num_embeddings=model_params.n_collision_values,
            embedding_dim=model_params.collision_embedding_size
        ) if model_params.collision_embedding_size > 0 else None

        encoder = get_encoder(
            name=model_params.encoder.type,
            in_channels=encoder_params.in_channels,
            depth=encoder_params.depth,
            weights=encoder_params.weights
        )

        fc = cls.create_fc_layers(
            encoder_output_size=cls.compute_output_size(encoder, encoder_params),
            embedding_size=model_params.collision_embedding_size + model_params.action_embedding_size,
            hidden_size=fc_params.hidden_size,
            output_size=fc_params.output_size,
            p_dropout=fc_params.p_dropout
        )

        return cls(encoder, fc, action_embedding, collision_embedding)

    @staticmethod
    def create_fc_layers(
            encoder_output_size: int,
            embedding_size: int,
            hidden_size: list,
            output_size: int,
            p_dropout: float = 0.0
    ):
        hidden_size.insert(0, encoder_output_size + embedding_size)

        layers = []
        for i in range(len(hidden_size)-1):
            layers += [
                DropoutPart(p_dropout, embedding_size) if i == 0 and embedding_size > 0 else nn.Dropout(p_dropout),
                nn.Linear(hidden_size[i], hidden_size[i+1]),
                nn.ReLU(True),
            ]

        return nn.Sequential(
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


class VONetV2(VONet):
    def forward(self, x, action=None, collision=None):
        x = self.encoder(x)
        x = self.flatten(x)
        if self.action_embedding:
            x = torch.cat([self.action_embedding(action).detach(), x], 1)
        if self.collision_embedding:
            x = torch.cat([self.collision_embedding(collision).detach(), x], 1)
        x = self.fc(x)

        return x

    @classmethod
    def from_config(cls, model_config):
        model_params = model_config.params
        encoder_params = model_params.encoder.params
        fc_params = model_params.fc.params

        action_embedding = nn.Embedding(
            num_embeddings=model_params.n_action_values,
            embedding_dim=model_params.action_embedding_size
        ) if model_params.action_embedding_size > 0 else None

        collision_embedding = nn.Embedding(
            num_embeddings=model_params.n_collision_values,
            embedding_dim=model_params.collision_embedding_size
        ) if model_params.collision_embedding_size > 0 else None

        backbone = getattr(resnet, model_params.encoder.type)(
            encoder_params.in_channels,
            encoder_params.base_planes,
            encoder_params.ngroups
        )

        encoder = nn.Sequential(
            Normalize(encoder_params.in_channels),
            backbone,
            nn.Conv2d(
                backbone.final_channels,
                encoder_params.num_compression_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, encoder_params.num_compression_channels),
            nn.ReLU(True),
        )

        fc = cls.create_fc_layers(
            encoder_output_size=cls.compute_output_size(encoder, encoder_params),
            embedding_size=model_params.collision_embedding_size + model_params.action_embedding_size,
            hidden_size=fc_params.hidden_size,
            output_size=fc_params.output_size,
            p_dropout=fc_params.p_dropout
        )

        return cls(encoder, fc, action_embedding, collision_embedding)


class VONetV3(VONetV2):
    def __init__(self,  encoder, fc, action_embedding=None, collision_embedding=None):
        super().__init__( encoder, fc, action_embedding, collision_embedding)
        self.fc = nn.ModuleList(self.fc)

    def forward(self, x, action=None, collision=None):
        x = self.encoder(x)
        x = self.flatten(x)
        for fc_layer in self.fc:
            if self.action_embedding:
                x = torch.cat([self.action_embedding(action).detach(), x], 1)
            if self.collision_embedding:
                x = torch.cat([self.collision_embedding(collision).detach(), x], 1)
            x = fc_layer(x)

        return x

    @staticmethod
    def create_fc_layers(
            encoder_output_size: int,
            embedding_size: int,
            hidden_size: list,
            output_size: int,
            p_dropout: float = 0.0
    ):
        hidden_size.insert(0, encoder_output_size)

        layers = []
        for i in range(len(hidden_size) - 1):
            layers.append(nn.Sequential(
                DropoutPart(p_dropout, embedding_size) if embedding_size > 0 else nn.Dropout(p_dropout),
                nn.Linear(hidden_size[i] + embedding_size, hidden_size[i + 1]),
                nn.ReLU(True),
            ))
        layers.append(nn.Sequential(
            DropoutPart(p_dropout, embedding_size) if embedding_size > 0 else nn.Dropout(p_dropout),
            nn.Linear(hidden_size[-1] + embedding_size, output_size)
        ))

        return layers
