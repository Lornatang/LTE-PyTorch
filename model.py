# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
from torch import Tensor
from torch import nn
import numpy as np
from typing import Any
from utils import make_coord
from torch.nn import functional as F_torch

__all__ = [

]


class LTE(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            encoder_channels: int = 64,
            out_channels: int = 3,
            channels: int = 256,
            encoder_arch: str = "edsr",
    ) -> None:
        super(LTE, self).__init__()
        if encoder_arch == "edsr":
            self.encoder = _EDSR(in_channels, encoder_channels)
        else:
            self.encoder = _EDSR(in_channels, encoder_channels)

        self.coefficient = nn.Conv2d(encoder_channels, channels, (3, 3), (1, 1), (1, 1))
        self.frequency = nn.Conv2d(encoder_channels, channels, (3, 3), (1, 1), (1, 1))
        self.phase = nn.Linear(2, channels // 2, bias=False)

        self.mlp = _MLP(channels, out_channels, [256, 256, 256])

        # Initialize all layer
        self._initialize_weights()

    def forward(self, x: Tensor, x_coord: Tensor, x_cell: Tensor = None) -> Tensor:
        return self._forward_impl(x, x_coord, x_cell)

    # Support torch.script function.
    def _forward_impl(self, x: Tensor, x_coord: Tensor, x_cell: Tensor) -> Tensor:
        temp_x = x
        device = x.device
        features_coord = make_coord(x.shape[-2:], flatten=False).to(device)
        features_coord = features_coord.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], 2, *x.shape[-2:])

        features = self.encoder(x)
        features_coefficient = self.coefficient(features)
        features_frequency = self.frequency(features)

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        # field radius (global: [-1, 1])
        rx = 2 / features.shape[-2] / 2
        ry = 2 / features.shape[-1] / 2

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                # prepare coefficient & frequency
                coord_ = x_coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_coefficient = F_torch.grid_sample(
                    input=features_coefficient,
                    grid=coord_.flip(-1).unsqueeze(1),
                    mode="nearest",
                    align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                q_frequency = F_torch.grid_sample(
                    input=features_frequency,
                    grid=coord_.flip(-1).unsqueeze(1),
                    mode="nearest",
                    align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                q_coord = F_torch.grid_sample(
                    input=features_coord,
                    grid=coord_.flip(-1).unsqueeze(1),
                    mode="nearest",
                    align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                rel_coord = x_coord - q_coord
                rel_coord[:, :, 0] *= features.shape[-2]
                rel_coord[:, :, 1] *= features.shape[-1]

                # prepare cell
                rel_cell = x_cell.clone()
                rel_cell[:, :, 0] *= features.shape[-2]
                rel_cell[:, :, 1] *= features.shape[-1]

                # basis generation
                batch_size, q = x_coord.shape[:2]
                q_frequency = torch.stack(torch.split(q_frequency, 2, dim=-1), dim=-1)
                q_frequency = torch.mul(q_frequency, rel_coord.unsqueeze(-1))
                q_frequency = torch.sum(q_frequency, dim=-2)
                q_frequency += self.phase(rel_cell.view((batch_size * q, -1))).view(batch_size, q, -1)
                q_frequency = torch.cat((torch.cos(np.pi * q_frequency), torch.sin(np.pi * q_frequency)), dim=-1)

                x = torch.mul(q_coefficient, q_frequency)

                pred = self.mlp(x.contiguous().view(batch_size * q, -1)).view(batch_size, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]
        areas[0] = areas[3]
        areas[3] = t
        t = areas[1]
        areas[1] = areas[2]
        areas[2] = t

        out = 0
        for pred, area in zip(preds, areas):
            out = out + pred * (area / tot_area).unsqueeze(-1)
        out += F_torch.grid_sample(
            input=temp_x,
            grid=x_coord.flip(-1).unsqueeze(1),
            mode="bilinear",
            padding_mode="border",
            align_corners=False)[:, :, 0, :].permute(0, 2, 1)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(_ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.rcb(x)

        out = torch.mul(out, 0.1)
        out = torch.add(out, identity)

        return out


class _EDSR(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 64,
            channels: int = 64,
            num_blocks: int = 16,
    ) -> None:
        super(_EDSR, self).__init__()
        # First layer
        self.conv1 = nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1))

        # Residual blocks
        trunk = []
        for _ in range(num_blocks):
            trunk.append(_ResidualConvBlock(channels))
        self.trunk = nn.Sequential(*trunk)

        # Second layer
        self.conv2 = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

    def forward(self, x: Tensor) -> Tensor:
        out1 = self.conv1(x)
        out = self.trunk(out1)
        out = self.conv2(out)
        out = torch.add(out, out1)

        return out


class _MLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels_list: list[int, int, int]):
        super(_MLP, self).__init__()
        layers = []

        last_channels = in_channels
        for hidden in hidden_channels_list:
            layers.append(nn.Linear(last_channels, hidden))
            layers.append(nn.ReLU(True))
            last_channels = hidden
        layers.append(nn.Linear(last_channels, out_channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape[:-1]
        out = self.layers(x.view(-1, x.shape[-1]))
        out = out.view(*shape, -1)

        return out


def lte_edsr(**kwargs: Any) -> LTE:
    model = LTE(encoder_arch="edsr", **kwargs)

    return model
