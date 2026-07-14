import os
import numpy as np
from torchvision import datasets, transforms
from PIL import Image

NUM_CLIENTS = 20
DATA_DIR = "./CIFAR_clients"

# Create CIFAR_clients directories
for i in range(1, NUM_CLIENTS + 1):
    os.makedirs(os.path.join(DATA_DIR, f"client_{i}", "images"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, f"client_{i}", "labels"), exist_ok=True)

# Download CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

# Convert images to numpy arrays for processing
images = np.array([
    np.transpose((img.numpy() * 255).astype(np.uint8), (1, 2, 0))
    for img, _ in dataset
])
labels = np.array(dataset.targets)

# IID split: equal number of samples per client
samples_per_client = len(dataset) // NUM_CLIENTS
for i in range(NUM_CLIENTS):
    start = i * samples_per_client
    end = (i + 1) * samples_per_client if i < NUM_CLIENTS - 1 else len(dataset)

    client_images = images[start:end]
    client_labels = labels[start:end]

    for idx, (img, lbl) in enumerate(zip(client_images, client_labels)):
        img_path = os.path.join(DATA_DIR, f"client_{i + 1}", "images", f"img_{idx}.png")
        label_path = os.path.join(DATA_DIR, f"client_{i + 1}", "labels", f"label_{idx}.txt")

        # Save image
        im = Image.fromarray(img.astype(np.uint8))
        im.save(img_path)

        # Save label
        with open(label_path, "w") as f:
            f.write(str(lbl))

print("Data split and saved for 20 CIFAR_clients successfully!")
