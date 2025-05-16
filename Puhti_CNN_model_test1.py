import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import sys

class Logger(object):
    def __init__(self, filename="test_results.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", buffering=1)  # line-buffered for real-time write

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("test_results.log")
sys.stderr = sys.stdout  # Optional: log errors as well


# === Configuration ===
NUM_CLASSES = 2
BATCH_SIZE = 32
TEST_DIR = r"C:\Users\satuh\OneDrive - Hämeen ammattikorkeakoulu (1)\2025 opinnot\Hamk\Machine learning\Assignment 4\Data for project\augmented_split_dataset\test"
MODEL_PATH = r"C:\Users\satuh\OneDrive - Hämeen ammattikorkeakoulu (1)\2025 opinnot\Hamk\Machine learning\Assignment 4\Codes\best_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Test Dataset and Loader ===
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Load Model ===
# === Load Model ===
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# Load checkpoint (modified from your original)
#checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
#model.load_state_dict(checkpoint['model_state_dict'])  # Extract just the model state
# With this:
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

model = model.to(DEVICE)
model.eval()

# === Evaluation ===
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# === Metrics ===
accuracy = accuracy_score(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds)
class_report = classification_report(all_labels, all_preds, target_names=test_dataset.classes)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
