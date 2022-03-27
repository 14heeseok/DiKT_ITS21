from einops.layers.torch import EinMix as Mix
from einops import rearrange, reduce
from torch import nn
import torch.nn.functional as F


def init(layer: Mix, scale=1.):
    layer.weight.data[:] = scale
    if layer.bias is not None:
        layer.bias.data[:] = 0
    return layer


def TokenMixer(num_features: int, n_patches: int,
               expansion_factor: int, dropout: float):
    n_hidden = n_patches * expansion_factor
    return nn.Sequential(
        nn.LayerNorm(num_features),
        Mix('b hw c -> b hid c', weight_shape='hw hid', bias_shape='hid',
            hw=n_patches, hid=n_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        Mix('b hid c -> b hw c', weight_shape='hid hw', bias_shape='hw',
            hw=n_patches, hid=n_hidden),
        nn.Dropout(dropout),
    )


def ChannelMixer(num_features: int, n_patches: int,
                 expansion_factor: int, dropout: float):
    n_hidden = num_features * expansion_factor
    return nn.Sequential(
        nn.LayerNorm(num_features),
        Mix('b h cw -> b h chid', weight_shape='cw chid', bias_shape='chid',
            cw=num_features, chid=n_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        Mix('b h chid -> b h cw', weight_shape='chid cw', bias_shape='cw',
            cw=num_features, chid=n_hidden),
        nn.Dropout(dropout),
    )


class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = Mix(
            "b t c-> b t0 c", weight_shape="t t0", bias_shape="t0",
            t=seq_len, t0=seq_len)
        init_eps = 1e-3/seq_len
        nn.init.constant_(self.spatial_proj.bias, 1.)
        nn.init.uniform_(self.spatial_proj.weight, -init_eps, init_eps)

    def forward(self, x):
        u, v = rearrange(x, "b t (c1 c2) ->c1 b t c2", c1=2)
        v = self.norm(v)
        v = self.spatial_proj(v)
        out = u * v
        return out


class MLPMixer_Layer(nn.Module):
    def __init__(self, h_dim, seq_size, expand,
                 dropout=0.0, layerscale_init=None):
        super().__init__()
        self.token_mixer = TokenMixer(
            h_dim, seq_size, expand, dropout)
        self.channel_mixer = ChannelMixer(
            h_dim, seq_size, expand, dropout)

    def forward(self, x):
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)
        return x


class ResMLP_Layer(nn.Module):
    def __init__(self, h_dim, seq_size: int, expand: int = 4,
                 dropout=None, layerscale_init=1.):
        super().__init__()
        self.branch_patches = nn.Sequential(
            init(Mix('b t c -> b t c', weight_shape='c', c=h_dim),
                 scale=layerscale_init),
            Mix('b t c -> b t0 c', weight_shape='t t0',
                bias_shape='t0', t=seq_size, t0=seq_size),
            init(Mix('b t c -> b t c', weight_shape='c',
                     bias_shape='c', c=h_dim)),
        )

        self.branch_channels = nn.Sequential(
            init(Mix('b t c -> b t c', weight_shape='c', c=h_dim),
                 scale=layerscale_init),
            nn.Linear(h_dim, expand * h_dim),
            nn.GELU(),
            nn.Linear(expand * h_dim, h_dim),
            init(Mix('b t c -> b t c', weight_shape='c',
                     bias_shape='c', c=h_dim)),
        )

    def forward(self, x):
        x = x + self.branch_patches(x)
        x = x + self.branch_channels(x)
        return x


class gMLP_Layer(nn.Module):
    def __init__(self, h_dim, seq_size: int, expand: int = 2,
                 dropout=None, layerscale_init=None):
        super().__init__()
        self.norm = nn.LayerNorm(h_dim)
        d_ffn = expand * h_dim
        self.channel_proj1 = nn.Linear(h_dim, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, h_dim)
        self.sgu = SpatialGatingUnit(d_ffn, seq_size)

    def forward(self, x):
        res = x
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        out = x + res
        return out


class MxKT(nn.Module):
    def __init__(self, mixer_type, question_num, num_layers, hidden_dim,
                 seq_size, scale, dropout=None, layerscale_init=None):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._question_num = question_num
        self._interaction_embedding = nn.Embedding(
            2*question_num + 1, hidden_dim)
        self._question_embedding = nn.Embedding(question_num, hidden_dim)
        if mixer_type == 'MLP':
            MixerLayer = MLPMixer_Layer
        elif mixer_type == 'ResMLP':
            MixerLayer = ResMLP_Layer
        elif mixer_type == 'gMLP':
            MixerLayer = gMLP_Layer
        else:
            raise NotImplementedError
        self._mixers = nn.Sequential(
            *[
                MixerLayer(hidden_dim, seq_size, scale, dropout,
                           layerscale_init) for _ in range(num_layers)
            ]
        )
        self._output_layer = nn.Linear(2 * hidden_dim, 1)

    def forward(self, interaction_id, target_id):
        x = self._interaction_embedding(interaction_id)
        x = self._mixers(x)
        x = reduce(x, ' b t hid -> b hid', reduction='mean')
        target = rearrange(self._question_embedding(target_id),
                           'b t hid -> b (t hid)')
        output = self._output_layer(
            rearrange([x, target], 'n b hid -> b (n hid)')
        )
        return output
