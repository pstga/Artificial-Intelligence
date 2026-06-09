import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class AddSpatialCoords(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, _, h, w = x.size()
        y_coords = torch.linspace(-1, 1, steps=h, device=x.device)
        y_coords = y_coords.view(1, 1, h, 1).expand(b, 1, h, w)
        x_coords = torch.linspace(-1, 1, steps=w, device=x.device)
        x_coords = x_coords.view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, y_coords, x_coords], dim=1)


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, max(in_channels // reduction, 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(in_channels // reduction, 4), in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AxialAttention(nn.Module):

    def __init__(self, channels, num_heads=4, max_len=128):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_w = nn.Linear(channels, channels * 3, bias=False)
        self.proj_w = nn.Linear(channels, channels, bias=False)
        self.norm_w = nn.LayerNorm(channels)
        self.rel_bias_w = nn.Embedding(2 * max_len - 1, num_heads)
        self.register_rel_index("rel_idx_w", max_len)

        self.qkv_h = nn.Linear(channels, channels * 3, bias=False)
        self.proj_h = nn.Linear(channels, channels, bias=False)
        self.norm_h = nn.LayerNorm(channels)

        self.rel_bias_h = nn.Embedding(2 * max_len - 1, num_heads)
        self.register_rel_index("rel_idx_h", max_len)
        nn.init.zeros_(self.proj_w.weight)
        nn.init.zeros_(self.proj_h.weight)

    def register_rel_index(self, name: str, max_len: int):
        positions = torch.arange(max_len)
        rel = positions.unsqueeze(0) - positions.unsqueeze(1)
        rel = rel + (max_len - 1)
        self.register_buffer(name, rel)

    def attn_1d(self, x_seq, qkv_layer, proj_layer, rel_bias_emb, rel_idx_buf):
        N, L, C = x_seq.shape
        H = self.num_heads
        D = self.head_dim

        qkv = qkv_layer(x_seq)
        qkv = qkv.reshape(N, L, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        rel_idx = rel_idx_buf[:L, :L]
        rel_idx = rel_idx.clamp(0, rel_bias_emb.num_embeddings - 1)
        bias = rel_bias_emb(rel_idx)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(N, L, C)
        return proj_layer(out)

    def forward(self, x):
        B, C, H, W = x.shape

        x_w = self.norm_w(x.permute(0, 2, 3, 1)).reshape(B * H, W, C)
        x_w_out = self.attn_1d(x_w, self.qkv_w, self.proj_w,
                                self.rel_bias_w, self.rel_idx_w)
        x = x + x_w_out.reshape(B, H, W, C).permute(0, 3, 1, 2)

        x_h = self.norm_h(x.permute(0, 2, 3, 1)).reshape(B * W, H, C)
        x_h_out = self.attn_1d(x_h, self.qkv_h, self.proj_h,
                                self.rel_bias_h, self.rel_idx_h)
        x = x + x_h_out.reshape(B, W, H, C).permute(0, 3, 2, 1)

        return x


class CompBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
        self.use_res = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * 4

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        ) if in_channels != hidden_dim else nn.Identity()

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.se = SEBlock(hidden_dim)

        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)

        if self.use_res:
            if self.drop_prob > 0.0 and self.training:
                out = drop_path(out, self.drop_prob)
            out += x
        return out


class SpectroModel(nn.Module):
    def __init__(self, num_classes=5, drop_path_rate=0.4, attn_heads=4, attn_max_len=128):
        super().__init__()
        self.coords = AddSpatialCoords()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        dpr = torch.linspace(0, drop_path_rate, 7).tolist()

        # revin la structura de blocuri din primul model dar cu straturi de atentie axiala
        self.blocks = nn.ModuleList([
            CompBlock(32,  64,  stride=1, drop_prob=dpr[0]),
            CompBlock(64,  64,  stride=1, drop_prob=dpr[1]),
            CompBlock(64,  128, stride=2, drop_prob=dpr[2]),
            CompBlock(128, 128, stride=1, drop_prob=dpr[3]),
            CompBlock(128, 256, stride=2, drop_prob=dpr[4]),  # block 4 — gets attention
            CompBlock(256, 256, stride=1, drop_prob=dpr[5]),  # block 5 — gets attention
            CompBlock(256, 512, stride=2, drop_prob=dpr[6])   # block 6 — gets attention
        ])

        # modulele de atentie axiala pentru ultimele blocuri din retea
        self.attn4 = AxialAttention(256, num_heads=attn_heads, max_len=attn_max_len)
        self.attn5 = AxialAttention(256, num_heads=attn_heads, max_len=attn_max_len)
        self.attn6 = AxialAttention(512, num_heads=attn_heads, max_len=attn_max_len)

        self.out_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # revenim la average pool simplu din primul model
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.coords(x)
        x = self.stem(x)

        for i, block in enumerate(self.blocks):
            x = block(x)
            # aplicam atentia axiala pe iesirea blocurilor 4, 5 si 6
            if i == 4:
                x = self.attn4(x)
            elif i == 5:
                x = self.attn5(x)
            elif i == 6:
                x = self.attn6(x)

        x = self.out_conv(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


class ModelEMA:
    def __init__(self, model, decay=0.99):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for param in self.ema.parameters():
            param.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ema_v, model_v in zip(self.ema.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(self.decay * ema_v + (1.0 - self.decay) * model_v)