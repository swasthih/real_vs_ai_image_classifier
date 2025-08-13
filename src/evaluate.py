import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.model import RealFakeCNN

# ================= Setup =================
start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ================= Load Test Data =================
test_dir = 'data/test'
test_data = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# ================= Load Model =================
model = RealFakeCNN().to(device)
model.load_state_dict(torch.load('model/real_fake_model.pth', map_location=device))
model.eval()

# ================= Inference =================
all_preds = []
all_labels = []
all_paths = [sample[0].split(test_dir + os.sep)[-1] for sample in test_loader.dataset.samples]

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ================= Save Predictions =================
df = pd.DataFrame({
    "filename": all_paths,
    "actual": [test_data.classes[i] for i in all_labels],
    "predicted": [test_data.classes[i] for i in all_preds]
})

os.makedirs("results", exist_ok=True)
df.to_csv("results/predictions.csv", index=False)
print("✅ Predictions saved to results/predictions.csv")

# ================= Classification Report =================
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=test_data.classes))

# ================= Confusion Matrix =================
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=test_data.classes, yticklabels=test_data.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
print("✅ Confusion matrix saved to results/confusion_matrix.png")

# ================= Time Report =================
end_time = time.time()
print(f"\n⏱️ Evaluation completed in {round(end_time - start_time, 2)} seconds")
