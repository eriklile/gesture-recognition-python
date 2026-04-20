import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

from models.mobilenet_v2_tsm import MobileNetV2TSM

NUM_CLASSES = 27
NUM_SEGMENTS = 8
IMG_SIZE = 112
BATCH_SIZE = 8
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class JesterDataset(Dataset):
    def __init__(self, csv_file, root_dir, num_segments=8, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.num_segments = num_segments
        self.transform = transform

    def _load_frames(self, folder):
        frame_files = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png"))
        ])

        if len(frame_files) == 0:
            raise ValueError(f"No frames found in {folder}")

        idxs = np.linspace(0, len(frame_files) - 1, self.num_segments).astype(int)
        frames = []
        for idx in idxs:
            img = cv2.imread(frame_files[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        return torch.stack(frames, dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_id = str(row["video_id"])
        label = int(row["label"])

        folder = os.path.join(self.root_dir, video_id)
        frames = self._load_frames(folder)
        return frames, label


def train():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    train_dataset = JesterDataset(
        csv_file="data/jester_metadata/train.csv",
        root_dir="data/raw",
        num_segments=NUM_SEGMENTS,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = MobileNetV2TSM(num_classes=NUM_CLASSES, num_segments=NUM_SEGMENTS).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for videos, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}: Loss={running_loss:.4f}, Acc={acc:.2f}%")

        torch.save(model.state_dict(), f"checkpoints/mobilenetv2_tsm_epoch{epoch+1}.pth")


if __name__ == "__main__":
    train()