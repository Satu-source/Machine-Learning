import os
import sys  # Added missing import
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models, utils
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Logger setup with correct filename
class Logger(object):
    def __init__(self, filename="output3.log"):  # Changed to output3.log
        self.terminal = sys.stdout
        self.log = open(filename, "a", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("/scratch/project_2014146/Satu_honka/Slurm_files/output3.log")  # Updated filename
sys.stderr = sys.stdout

# === Configuration ===
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
NUM_CLASSES = 2
DATA_DIR = "file_path"
SAVE_PATH = "file_path"  # Updated filename
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Data Augmentation ===
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === Datasets and Dataloaders ===
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_transform)

# Handle class imbalance
class_counts = np.array([len([x for x in train_dataset.targets if x == c]) for c in range(NUM_CLASSES)])
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
samples_weights = class_weights[train_dataset.targets]
sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# === Model ===
model = models.resnet50(pretrained=True)

# Freeze layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last layers
for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, NUM_CLASSES)
)
model = model.to(DEVICE)

# === Training Setup ===
criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
scaler = torch.cuda.amp.GradScaler()
writer = SummaryWriter(log_dir="runs/experiment_resnet50_v3")  # Updated run name
best_val_acc = 0.0
patience = 5
epochs_no_improve = 0

torch.backends.cudnn.benchmark = True

# === Training Loop ===
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_start = time.time()
    running_loss = 0.0
    correct = 0
    total = 0

    train_bar = tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{NUM_EPOCHS}', leave=False)
    for images, labels in train_bar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        train_bar.set_postfix({
            'loss': loss.item(),
            'acc': f"{(preds == labels).sum().item() / labels.size(0):.2f}"
        })

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        val_bar = tqdm(val_loader, desc='Validation', leave=False)
        for images, labels in val_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    train_loss = running_loss / total
    train_acc = correct / total
    val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    scheduler.step()

    # Logging
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)

    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=train_dataset.classes,
                yticklabels=train_dataset.classes)
    plt.title("Confusion Matrix")
    plt.savefig(f'confusion_matrix_epoch_{epoch+1}.png')
    plt.close()

    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc
        }, SAVE_PATH)
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs!")
            break

    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
          f"Time: {time.time() - epoch_start:.2f}s")

# Save final model with correct filename
torch.save(model.state_dict(), "final_model3.pt")  # Updated filename
print(f"\nTraining complete. Best Val Accuracy: {best_val_acc:.4f}")
writer.close()
