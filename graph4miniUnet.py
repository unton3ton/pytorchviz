import torch
import torch.nn as nn
from torchviz import make_dot

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MiniUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)  # 32 -> 16
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)  # 16 -> 8

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.up2 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)  # 8 -> 16
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 32 (from up) + 32 (skip) = 64
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)  # 16 -> 32
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 16 + 16 = 32
            nn.ReLU(),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)           # [B, 16, 32, 32]
        p1 = self.pool1(e1)         # [B, 16, 16, 16]

        e2 = self.enc2(p1)          # [B, 32, 16, 16]
        p2 = self.pool2(e2)         # [B, 32, 8, 8]

        # Bottleneck
        b = self.bottleneck(p2)     # [B, 32, 8, 8]

        # Decoder
        u2 = self.up2(b)            # [B, 32, 16, 16]
        u2 = torch.cat([u2, e2], dim=1)  # skip connection → [B, 64, 16, 16]
        d2 = self.dec2(u2)          # [B, 16, 16, 16]

        u1 = self.up1(d2)           # [B, 16, 32, 32]
        u1 = torch.cat([u1, e1], dim=1)  # skip connection → [B, 32, 32, 32]
        d1 = self.dec1(u1)          # [B, 1, 32, 32]

        return d1

# Создаём модель и данные
model = MiniUNet(in_channels=3, out_channels=1).to(device)
x = torch.randn(1, 3, 32, 32).to(device)  # batch=1, RGB, 32x32

# Прямой проход
y = model(x)

# Визуализация
dot = make_dot(y.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)

# Сохраняем
dot.render("mini_unet_graph", format="png", cleanup=True)
print("Граф U-Net сохранён как 'mini_unet_graph.png'")