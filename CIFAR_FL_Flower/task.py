"""Adapted from flower tutorial: A Flower / PyTorch app."""
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class Net(nn.Module):
    """A simple 3D CNN model adapted for volume data."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CIFARDataset(Dataset):
    """CIFAR-10 dataset for a single client, loaded from disk with optional label poisoning."""

    def __init__(self, data_dir, transform=None, poisoned=False,
                 flip_percent="050", poisoned_clients=None, client_id=None):
        """
        Args:
            data_dir (str): Path to the client folder containing `images/` and `labels/`
            transform: torchvision transforms to apply
            poisoned (bool): whether to apply label flipping
            flip_percent (str): percentage of labels to flip
            poisoned_clients (list[int]): list of poisoned client IDs
            client_id (int): current client ID (0-indexed)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.poisoned = poisoned
        self.client_id = client_id
        self.flip_percent = flip_percent
        self.poisoned_clients = poisoned_clients if poisoned_clients else []

        # Determine if this client is poisoned
        self.is_poisoned_client = poisoned and (self.client_id in self.poisoned_clients)
        if self.is_poisoned_client:
            flipped_labels_dir = Path(
                "flipped_labels") / f"{flip_percent}percent_flip" / f"client_{client_id}" / "labels"
            label_paths = sorted(list(flipped_labels_dir.glob("*.txt")))
            print(
                f"⚠️ Client {self.client_id} (client_{client_id}) is poisoned - loading from {flipped_labels_dir}")
        else:
            # Load from original CIFAR_clients directory
            label_paths = sorted(list((self.data_dir / "labels").glob("*.txt")))

        # Load all image paths
        self.image_paths = sorted(list((self.data_dir / "images").glob("*.png")))
        # Load all labels
        self.labels = []
        for label_path in label_paths:
            with open(label_path, "r") as f:
                self.labels.append(int(f.read().strip()))
        self.labels = np.array(self.labels)

        # Verify we have matching number of images and labels
        if len(self.image_paths) != len(self.labels):
            raise ValueError(
                f"Mismatch: {len(self.image_paths)} images but {len(self.labels)} labels "
                f"for client {self.client_id}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label

def load_data(partition_id: int, batch_size: int = 32, poisoned: bool = False, # Increased batch_size
              flip_percent: str = "000", data_root: str = "CIFAR_clients", poisoned_clients=None):
    """Load data for a specific client with optional poisoning.

    Args:
        partition_id: Client ID (0-indexed, so client_1 is partition_id=0)
        batch_size: Batch size for dataloaders
        poisoned: If True, load poisoned labels from flipped_labels directory
        flip_percent: Flip percentage ('010', '025', '050', '075', '100')
        data_root: Root directory containing client data
        poisoned_clients: ID of poisoned CIFAR_clients

    Returns:
        trainloader, valloader
    """
    data_dir = f"{data_root}/client_{partition_id}"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize
    ])
    dataset = CIFARDataset(data_dir, poisoned=poisoned,
                           flip_percent=flip_percent, poisoned_clients=poisoned_clients,
                           client_id=partition_id, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    trainloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        persistent_workers=False # Disable persistent workers
    )
    valloader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        persistent_workers=False # Disable persistent workers
    )
    return trainloader, valloader

def train(model, trainloader, epochs, lr, device):
    print(f"Starting training on {device} | Epochs: {epochs} | LR: {lr}")
    print(f"Total batches per epoch: {len(trainloader)}")
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    optimizer = optim.SGD(model.parameters(), lr=lr)
    batch_idx = 0
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_idx += 1

        epoch_duration = time.time() - epoch_start_time
        print(f"  Epoch {epoch + 1}/{epochs} completed in {epoch_duration:.2f}s")

    avg_loss = total_loss / batch_idx if batch_idx > 0 else 0.0
    total_duration = time.time() - start_time
    print(f"Training complete in {total_duration:.2f}s - Final Avg loss: {avg_loss:.4f}")
    return avg_loss


def test(model, testloader, device):
    model.eval().to(device)
    criterion = nn.CrossEntropyLoss()
    correct, total = 0, 0
    avg_loss = 0.0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            loss = criterion(outputs, target).item()
            correct += (predicted == target).sum().item()
            avg_loss += loss
    accuracy = correct / total
    avg_loss = avg_loss/len(testloader)
    return avg_loss, accuracy

class MetricsLogger:
    """Logger for FL metrics collection."""

    def __init__(self, poisoned: bool = False, flip_percent: str = None,
                 pid_enabled: bool = False, pid_params: dict = None):
        self.poisoned = poisoned
        self.client_losses, self.client_accuracies = [], []
        self.avg_losses, self.avg_accuracies = [], []

        #PID
        self.pid_scores = []
        self.exclusions = []
        self.pid_enabled = pid_enabled
        self.pid_params = pid_params or {}

        if pid_enabled:
            pid_str = f"_kp{pid_params.get('kp', 1.0)}_ki{pid_params.get('ki', 0.05)}_kd{pid_params.get('kd', 0.5)}_th{pid_params.get('threshold', 2.0)}"
            self.output_dir = f"results/attack/{flip_percent}flipped{pid_str}"
        else:
            self.output_dir = f"results/attack/{flip_percent}flipped" if poisoned else "results/no_attack"

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)



    def log_client_metrics(self, round_num, client_id, loss, accuracy):
        self.client_losses.append({"round": round_num, "client_id": client_id, "loss": loss})
        self.client_accuracies.append({"round": round_num, "client_id": client_id, "accuracy": accuracy})

    def log_average_metrics(self, round_num, avg_loss, avg_accuracy):
        self.avg_losses.append({"round": round_num, "avg_loss": avg_loss})
        self.avg_accuracies.append({"round": round_num, "avg_accuracy": avg_accuracy})

    def log_pid_score(self, round_num, client_id, pid_score, distance):
        """Log PID score and distance for a client."""
        self.pid_scores.append({
            "round": round_num,
            "client_id": client_id,
            "pid_score": pid_score,
            "distance": distance
        })

    def log_exclusion(self, round_num, client_id, excluded, pid_score):
        """Log client exclusion status."""
        self.exclusions.append({
            "round": round_num,
            "client_id": client_id,
            "excluded": int(excluded),
            "pid_score": pid_score
        })

    def save_to_csv(self):
        def _save(path, data, fields):
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(data)
        _save(f"{self.output_dir}/client_losses.csv", self.client_losses, ["round", "client_id", "loss"])
        _save(f"{self.output_dir}/client_accuracies.csv", self.client_accuracies, ["round", "client_id", "accuracy"])
        _save(f"{self.output_dir}/avg_losses.csv", self.avg_losses, ["round", "avg_loss"])
        _save(f"{self.output_dir}/avg_accuracies.csv", self.avg_accuracies, ["round", "avg_accuracy"])

        if self.pid_scores:
            _save(f"{self.output_dir}/pid_scores.csv", self.pid_scores,
                  ["round", "client_id", "pid_score", "distance"])
        if self.exclusions:
            _save(f"{self.output_dir}/exclusions.csv", self.exclusions,
                  ["round", "client_id", "excluded", "pid_score"])
        print(f"Metrics saved to {self.output_dir}/")