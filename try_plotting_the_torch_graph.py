# sudo apt update
# sudo apt install graphviz
# pip install torchviz

# MiniResNet с параллельными путями

import torch
import torch.nn as nn
from torchviz import make_dot

# Устройство
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class InterestingNet(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=1):
        super().__init__()
        # Общий начальный слой
        self.stem = nn.Linear(input_dim, hidden_dim)
        
        # Ветвь 1: последовательная
        self.branch1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Ветвь 2: с нелинейностью и dropout
        self.branch2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Объединение
        self.fuse = nn.Linear(hidden_dim, output_dim)  # hidden_dim//2 + hidden_dim//2 = hidden_dim

    def forward(self, x):
        x = torch.relu(self.stem(x))
        
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        
        # Склеиваем по признакам
        combined = torch.cat([out1, out2], dim=1)
        
        # Добавим residual connection от x (если размеры совпадают — иначе пропустим)
        # В нашем случае x: [B, 32], combined: [B, 32] → можно!
        if x.shape == combined.shape:
            combined = combined + x  # skip connection
        
        out = self.fuse(combined)
        return out

# Создаём модель и данные
model = InterestingNet(input_dim=10, hidden_dim=32, output_dim=1).to(device)
x = torch.randn(2, 10).to(device)  # batch_size=2 для наглядности

# Прямой проход
y = model(x)

# Визуализация графа
dot = make_dot(y.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)

# Сохраняем
dot.render("interesting_net", format="png", cleanup=True)
print("Граф сохранён как 'interesting_net.png'")

# ИЛИ открыть в просмотрщике (если поддерживается ОС)
# dot.view()