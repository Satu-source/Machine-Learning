import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models, utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Making output.log file from version 2
import sys

class Logger(object):
    def __init__(self, filename="output2.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", buffering=1)  # line-buffered for real-time write

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("output2.log")
sys.stderr = sys.stdout  # Optional: log errors as well




# === Configuration ===
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
NUM_CLASSES = 2
DATA_DIR = "File path where data is"
SAVE_PATH = "best_model2.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Datasets and Dataloaders ===
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# === Model ===
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = torch.nn.DataParallel(model)
model = model.to(DEVICE)

# Verify That Multiple GPUs Are Being Used
print(f"Using {torch.cuda.device_count()} GPUs")

# === Optimizer, Loss, Scheduler ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True)

# === Mixed Precision ===
scaler = torch.cuda.amp.GradScaler()

# === TensorBoard ===
writer = SummaryWriter(log_dir="runs/experiment_resnet50")

# === Training Loop ===
best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_start = time.time()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    for images, labels in progress_bar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        progress_bar.set_postfix(loss=loss.item())

    train_acc = correct / total
    avg_train_loss = running_loss / total

    # === Validation ===
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = correct / total
    avg_val_loss = val_loss / total

    # === TensorBoard Logs ===
    writer.add_scalar("Loss/train", avg_train_loss, epoch)
    writer.add_scalar("Loss/val", avg_val_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)

    # === Show Classification Report ===
    print("\nValidation Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

    # === Confusion Matrix ===
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    # === Log some wrong predictions ===
    wrong = [(x, y, p) for x, y, p in zip(images, labels, preds) if y != p]
    if wrong:
        wrong_images = torch.stack([item[0].cpu() for item in wrong[:8]])
        grid = utils.make_grid(wrong_images, nrow=4)
        writer.add_image("Wrong Predictions", grid, epoch)

    # === Learning Rate Adjustment ===
    scheduler.step(val_acc)

    # === Save Best Model (Checkpointing) ===
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc
        }, SAVE_PATH)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
          f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} | "
          f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | "
          f"Time: {time.time() - epoch_start:.2f}s")

# Save final model
torch.save(model.state_dict(), "final_model2.pt")
print("Training complete. Best Val Accuracy:", best_val_acc)
