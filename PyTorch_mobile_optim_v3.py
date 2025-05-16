import os
import time
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.mobile_optimizer import optimize_for_mobile
from sklearn.metrics import accuracy_score

# === Paths ===
MODEL_PATH = r"C:\Users\satuh\OneDrive - Hämeen ammattikorkeakoulu (1)\2025 opinnot\Hamk\Machine learning\Assignment 4\Codes\best_model.pt"
QUANTIZED_PATH = r"C:\Users\satuh\OneDrive - Hämeen ammattikorkeakoulu (1)\2025 opinnot\Hamk\Machine learning\Assignment 4\Codes\model_quantized.pt"
SCRIPTED_PATH = "model_scripted.pt"
OPTIMIZED_PATH = "model_optimized.ptl"
TEST_DIR = r"C:\Users\satuh\OneDrive - Hämeen ammattikorkeakoulu (1)\2025 opinnot\Hamk\Machine learning\Assignment 4\Data for project\augmented_split_dataset\test"
DEVICE = torch.device("cpu")

# === Transforms & DataLoader ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# === Load Trained Model ===
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# === Evaluate Original Accuracy ===
def evaluate(model):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    return accuracy_score(all_labels, all_preds)

original_acc = evaluate(model)
print(f"Original Model Accuracy: {original_acc * 100:.2f}%")


# === Quantize Model ===
quantized_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
torch.save(quantized_model, QUANTIZED_PATH)
print("Quantized model saved.")

# === TorchScript Export ===
dummy_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(quantized_model, dummy_input)
traced_model.save(SCRIPTED_PATH)
print(f"TorchScript model saved: {SCRIPTED_PATH}")

# === Optimize for Mobile ===
optimized_model = optimize_for_mobile(traced_model)
optimized_model.save(OPTIMIZED_PATH)
print(f"Optimized model saved for Android: {OPTIMIZED_PATH}")

# === Performance Info ===

# === Evaluate Optimized Accuracy ===
optimized_acc = evaluate(quantized_model)
print(f"Optimized Model Accuracy: {optimized_acc * 100:.2f}%")

# === Accuracy Drop ===
drop = original_acc - optimized_acc
print(f"Accuracy Drop: {drop * 100:.2f}%")

# Warm-up
for _ in range(10):
    _ = optimized_model(dummy_input)

# Latency measurement
latencies = []
for _ in range(50):
    start = time.time()
    _ = optimized_model(dummy_input)
    end = time.time()
    latencies.append((end - start) * 1000)  # in milliseconds

avg_latency = sum(latencies) / len(latencies)
model_size_mb = os.path.getsize(OPTIMIZED_PATH) / 1e6

# === Summary ===
print("\n=== Performance Summary ===")
print(f"Optimized Model Size: {model_size_mb:.2f} MB")
print(f"Avg Latency (CPU): {avg_latency:.2f} ms")
print(f"Original Accuracy: {original_acc * 100:.2f}%")
print(f"Optimized Accuracy: {optimized_acc * 100:.2f}%")
print(f"Accuracy Drop: {drop * 100:.2f}%")
print("Model is ready for Android Studio (.ptl)")
