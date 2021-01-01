import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Flatten, CategoricalNet
from . import resnet50

import numpy as np


class Policy(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size=512,
        num_recurrent_layers=1,
        blind=0,
        rnn_type="GRU",
        goal_sensor_uuid="pointgoal"
    ):
        super().__init__()
        self.dim_actions = action_space.n

        self.net = Net(
            observation_space=observation_space,
            hidden_size=hidden_size,
            num_recurrent_layers=num_recurrent_layers,
            blind=blind,
            rnn_type=rnn_type,
            goal_sensor_uuid=goal_sensor_uuid,
        )

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )

    def forward(self, *x):
        return None

    def act(
        self, observations, rnn_hidden_states, prev_actions, masks, deterministic=False
    ):
        value, actor_features, rnn_hidden_states, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(actor_features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        value, _, _, _ = self.net(observations, rnn_hidden_states, prev_actions, masks)
        return value

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        value, actor_features, rnn_hidden_states, cnn_feats = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(actor_features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return (value, action_log_probs, distribution_entropy, rnn_hidden_states)


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        input_channels=3,
        baseplanes=32,
        ngroups=32,
        spatial_size=128,
        flat_output_size=2048,
        make_backbone=resnet50,
    ):
        super().__init__()

        self.backbone = make_backbone(input_channels, baseplanes, ngroups)

        final_spatial = int(spatial_size * self.backbone.final_spatial_compress)
        bn_size = int(round(flat_output_size / (final_spatial ** 2)))
        self.output_size = (bn_size, final_spatial, final_spatial)

        self.bn = nn.Sequential(
            nn.Conv2d(
                self.backbone.final_channels,
                bn_size,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, bn_size),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.backbone(x)

        return self.bn(x)


class Net(nn.Module):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space,
        hidden_size,
        num_recurrent_layers,
        blind,
        rnn_type,
        goal_sensor_uuid="pointgoal"
    ):
        super().__init__()

        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
            self._n_input_rgb = 3
            self.register_buffer(
                "grayscale_kernel",
                torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32).view(
                    1, 3, 1, 1
                ),
            )
            spatial_size = observation_space.spaces["rgb"].shape[0] // 2
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
            spatial_size = observation_space.spaces["depth"].shape[0] // 2
        else:
            self._n_input_depth = 0

        self.prev_action_embedding = nn.Embedding(5, 32)
        self._n_prev_action = 32

        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_input_goal = observation_space.spaces[self.goal_sensor_uuid].shape[0]
        self._old_goal_format = False
        if self._old_goal_format:
            self._n_input_goal -= 1
        self.tgt_embed = nn.Linear(self._n_input_goal, 32)
        self._n_input_goal = 32

        self._hidden_size = hidden_size

        rnn_input_size = self._n_input_goal + self._n_prev_action
        if not blind:
            assert self._n_input_depth + self._n_input_rgb > 0
            encoder = ResNetEncoder(
                self._n_input_depth + self._n_input_rgb, 32, 16, spatial_size
            )
            self.cnn = nn.Sequential(
                encoder,
                Flatten(),
                nn.Linear(np.prod(encoder.output_size), hidden_size),
                nn.ReLU(True),
            )
            rnn_input_size += self._hidden_size
        else:
            self._n_input_rgb = 0
            self._n_input_depth = 0
            self.cnn = None

        self._rnn_type = rnn_type
        self._num_recurrent_layers = num_recurrent_layers
        self.rnn = getattr(nn, rnn_type)(
            rnn_input_size, hidden_size, num_layers=num_recurrent_layers
        )
        self.critic_linear = nn.Linear(hidden_size, 1)

        self.layer_init()
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def num_recurrent_layers(self):
        return self._num_recurrent_layers * (2 if "LSTM" in self._rnn_type else 1)

    def layer_init(self):
        if self.cnn is not None:
            for layer in self.cnn:
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    nn.init.orthogonal_(layer.weight, nn.init.calculate_gain("relu"))
                    nn.init.constant_(layer.bias, val=0)

        #  for name, param in self.rnn.named_parameters():
        #  if "weight" in name:
        #  nn.init.orthogonal_(param)
        #  elif "bias" in name:
        #  nn.init.constant_(param, 0)

        nn.init.orthogonal_(self.critic_linear.weight, gain=1)
        nn.init.constant_(self.critic_linear.bias, val=0)

    def _pack_hidden(self, hidden_states):
        if "LSTM" in self._rnn_type:
            hidden_states = torch.cat([hidden_states[0], hidden_states[1]], dim=0)

        return hidden_states

    def _unpack_hidden(self, hidden_states):
        if "LSTM" in self._rnn_type:
            hidden_states = (
                hidden_states[0 : self._num_recurrent_layers],
                hidden_states[self._num_recurrent_layers :],
            )

        return hidden_states

    def _mask_hidden(self, hidden_states, masks):
        if isinstance(hidden_states, tuple):
            hidden_states = tuple(v * masks for v in hidden_states)
        else:
            hidden_states = masks * hidden_states

        return hidden_states

    def forward_rnn(self, x, hidden_states, masks):
        if x.size(0) == hidden_states.size(1):
            hidden_states = self._unpack_hidden(hidden_states)
            x, hidden_states = self.rnn(
                x.unsqueeze(0), self._mask_hidden(hidden_states, masks.unsqueeze(0))
            )
            x = x.squeeze(0)
        else:
            # x is a (T, N, -1) tensor flattened to (T * N, -1)
            n = hidden_states.size(1)
            t = int(x.size(0) / n)

            # unflatten
            x = x.view(t, n, x.size(1))
            masks = masks.view(t, n)

            # steps in sequence which have zero for any agent. Assume t=0 has
            # a zero in it.
            has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                has_zeros = [has_zeros.item() + 1]  # handle scalar
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [t]

            hidden_states = self._unpack_hidden(hidden_states)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # process steps that don't have any zeros in masks together
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hidden_states = self.rnn(
                    x[start_idx:end_idx],
                    self._mask_hidden(hidden_states, masks[start_idx].view(1, -1, 1)),
                )

                outputs.append(rnn_scores)

            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            x = x.view(t * n, -1)  # flatten

        hidden_states = self._pack_hidden(hidden_states)
        return x, hidden_states

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            #  rgb_observations = F.conv2d(
            #  rgb_observations, self.grayscale_kernel
            #  )

            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]

            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)

            cnn_input.append(depth_observations)

        goal_observations = observations[self.goal_sensor_uuid]
        if self._old_goal_format:
            rho_obs = goal_observations[:, 0].clone()
            phi_obs = -torch.atan2(goal_observations[:, 2], goal_observations[:, 1])
            goal_observations = torch.stack([rho_obs, phi_obs], -1)

        goal_observations = self.tgt_embed(goal_observations)
        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().squeeze(-1)
        )

        x = []
        cnn_feats = None
        if len(cnn_input) > 0:
            cnn_input = torch.cat(cnn_input, dim=1)
            cnn_input = F.avg_pool2d(cnn_input, 2)
            cnn_feats = self.cnn(cnn_input)
            x += [cnn_feats]

        x += [goal_observations, prev_actions]

        x = torch.cat(x, dim=1)  # concatenate goal vector
        x, rnn_hidden_states = self.forward_rnn(x, rnn_hidden_states, masks)

        return self.critic_linear(x), x, rnn_hidden_states, cnn_feats