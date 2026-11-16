"""CNN inference for real-vs-AI image classification.

The architecture, input size (128x128) and normalisation (mean=std=0.5) are
identical to the training / test notebooks.  Label 0 = Real, label 1 = AI.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageOps

MODEL_DIR = Path(__file__).parent / "models"


class CNN(nn.Module):
    """Three conv blocks (Conv-BN-ReLU-MaxPool) followed by a small FC head."""

    def __init__(self, num_filters=32, dropout=0.3, img_size=128):
        super().__init__()
        f = num_filters
        self.features = nn.Sequential(
            nn.Conv2d(3, f, 3, padding=1), nn.BatchNorm2d(f), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(f, f * 2, 3, padding=1), nn.BatchNorm2d(f * 2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(f * 2, f * 4, 3, padding=1), nn.BatchNorm2d(f * 4), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        flat = f * 4 * (img_size // 8) ** 2
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, 256), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout), nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class Detector:
    def __init__(self, model_dir=MODEL_DIR, device=None):
        model_dir = Path(model_dir)
        self.cfg = json.loads((model_dir / "cnn_config.json").read_text())
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = CNN(self.cfg["num_filters"], self.cfg["dropout"], self.cfg["img_size"])
        state = torch.load(model_dir / "cnn_best.pth", map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        size = self.cfg["img_size"]
        self.transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(self.cfg["norm_mean"], self.cfg["norm_std"]),
        ])

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        """Return {'label', 'p_ai', 'p_real', 'confidence'} for a PIL image."""
        image = ImageOps.exif_transpose(image).convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)
        p_real, p_ai = torch.softmax(self.model(x), dim=1)[0].tolist()
        is_ai = p_ai >= 0.5
        return {
            "label": "AI-generated" if is_ai else "Real",
            "p_ai": p_ai,
            "p_real": p_real,
            "confidence": p_ai if is_ai else p_real,
        }
