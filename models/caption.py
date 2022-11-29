import torch
from torch import nn
import torch.nn.functional as F

from .utils import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone
from .transformer import build_transformer


class Caption(nn.Module):

    def __init__(self, backbone, transformer, hidden_dim, vocab_size):
        super().__init__()
        self.backbone = backbone
        self.input_proj = nn.Conv2d(in_channels=backbone.num_channels,
                                    out_channels=hidden_dim,
                                    kernel_size=1)
        self.transformer = transformer
        self.mlp = MLP(hidden_dim, 512, vocab_size, 3)

    def forward(self, samples, target, target_mask):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        src = self.input_proj(src)
        pos = pos[-1]
        assert mask is not None

        # flatten vectors
        src = src.flatten(2)
        pos = pos.flatten(2)
        mask = mask.flatten(1)

        hs = self.transformer(src, mask, pos, target,
                              target_mask)
        out = self.mlp(hs.permute(1, 0, 2))
        return out


class CaptionGlobal(nn.Module):

    # TODO separate backbones / position encodings for local and global features?

    def __init__(self, backbone, transformer, hidden_dim, vocab_size):
        super().__init__()
        self.backbone = backbone
        self.input_proj = nn.Conv2d(in_channels=backbone.num_channels,
                                    out_channels=hidden_dim,
                                    kernel_size=1)
        self.transformer = transformer
        self.mlp = MLP(hidden_dim, 512, vocab_size, 3)

    def forward(self, t_samples, g_samples, target, target_mask):

        # target features

        if not isinstance(t_samples, NestedTensor):
            t_samples = nested_tensor_from_tensor_list(t_samples)
        t_features, t_pos = self.backbone(t_samples)
        t_src, t_mask = t_features[-1].decompose()
        t_src = self.input_proj(t_src)
        t_pos = t_pos[-1]
        assert t_mask is not None
        # flatten vectors
        t_src = t_src.flatten(2)
        t_pos = t_pos.flatten(2)
        t_mask = t_mask.flatten(1)

        # global features

        if not isinstance(g_samples, NestedTensor):
            g_samples = nested_tensor_from_tensor_list(g_samples)
        g_features, g_pos = self.backbone(g_samples)
        g_src, g_mask = g_features[-1].decompose()
        g_src = self.input_proj(g_src)
        g_pos = g_pos[-1]
        assert g_mask is not None
        # flatten vectors
        g_src = g_src.flatten(2)
        g_pos = g_pos.flatten(2)
        g_mask = g_mask.flatten(1)

        # concatenate

        src = torch.concat([t_src, g_src], 2)
        pos = torch.concat([t_pos, g_pos], 2)
        mask = torch.concat([t_mask, g_mask], 1)

        hs = self.transformer(src, mask, pos, target, target_mask)
        out = self.mlp(hs.permute(1, 0, 2))
        return out


class CaptionLoc(nn.Module):

    def __init__(self, backbone, transformer, hidden_dim, vocab_size):
        super().__init__()
        self.backbone = backbone
        self.input_proj = nn.Conv2d(in_channels=backbone.num_channels,
                                    out_channels=hidden_dim,
                                    kernel_size=1)
        self.loc_proj = nn.Linear(7, hidden_dim)
        self.transformer = transformer
        self.mlp = MLP(hidden_dim, 512, vocab_size, 3)

    def forward(self, t_samples, loc_feats, target, target_mask):

        # target features
        if not isinstance(t_samples, NestedTensor):
            t_samples = nested_tensor_from_tensor_list(t_samples)
        t_features, t_pos = self.backbone(t_samples)
        t_src, t_mask = t_features[-1].decompose()
        t_src = self.input_proj(t_src)
        t_pos = t_pos[-1]
        assert t_mask is not None
        # flatten vectors
        t_src = t_src.flatten(2)
        t_pos = t_pos.flatten(2)
        t_mask = t_mask.flatten(1)

        # location features

        loc_src = self.loc_proj(loc_feats).unsqueeze(-1)
        loc_pos = torch.ones_like(loc_src).unsqueeze(-1)
        loc_masks = torch.zeros((loc_feats.shape[0], 1)).bool()

        # concatenate
        src = torch.concat([t_src, loc_src], 2)
        pos = torch.concat([t_pos, loc_pos], 2)
        mask = torch.concat([t_mask, loc_masks], 1)

        hs = self.transformer(src, mask, pos, target, target_mask)
        out = self.mlp(hs.permute(1, 0, 2))
        return out


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_model(config):
    backbone = build_backbone(config)
    transformer = build_transformer(config)

    use_global = config.use_global_features
    use_location = config.use_location_features

    print(f'global features: {use_global}, location features: {use_location}')

    if use_global and not use_location:
        # global features
        model = CaptionGlobal(backbone, transformer, config.hidden_dim,
                              config.vocab_size)
    elif not use_global and use_location:
        # loc features
        model = CaptionLoc(backbone, transformer, config.hidden_dim,
                        config.vocab_size)
    elif use_global and use_location:
        # both global image and loc features
        raise NotImplementedError()
    else:
        # default / no global image or loc features
        model = Caption(backbone, transformer, config.hidden_dim,
                        config.vocab_size)

    criterion = torch.nn.CrossEntropyLoss()

    return model, criterion