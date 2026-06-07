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
    output = x.div(keep_prob) * random_tensor
    return output


class AddSpatialCoords(nn.Module):
    """Acum adăugăm atât axa Frecvenței (Y) cât și axa Timpului (X)"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, _, h, w = x.size()
        # Coordonate Y (Frecvență)
        y_coords = torch.linspace(-1, 1, steps=h, device=x.device)
        y_coords = y_coords.view(1, 1, h, 1).expand(b, 1, h, w)
        # Coordonate X (Timp)
        x_coords = torch.linspace(-1, 1, steps=w, device=x.device)
        x_coords = x_coords.view(1, 1, 1, w).expand(b, 1, h, w)

        # Concat: Acum avem 3 canale (Grayscale + Y + X)
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


class LiteBlock(nn.Module):
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


class SpectroNeXt(nn.Module):
    def __init__(self, num_classes=5, drop_path_rate=0.4):
        super().__init__()
        self.coords = AddSpatialCoords()

        # SCHIMBAT LA 3 CANALE (din cauza coordonatelor 2D de mai sus)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 7)]

        self.blocks = nn.ModuleList([
            LiteBlock(32, 64, stride=1, drop_prob=dpr[0]),  # was 32→64, now 32→32
            LiteBlock(64, 64, stride=1, drop_prob=dpr[1]),  # was 64→64, now 32→64  ← mismatch stops here in log
            LiteBlock(64, 128, stride=2, drop_prob=dpr[2]),
            LiteBlock(128, 128, stride=1, drop_prob=dpr[3]),
            LiteBlock(128, 256, stride=2, drop_prob=dpr[4]),
            LiteBlock(256, 256, stride=1, drop_prob=dpr[5]),
            LiteBlock(256, 512, stride=2, drop_prob=dpr[6])
        ])

        self.out_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.coords(x)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
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