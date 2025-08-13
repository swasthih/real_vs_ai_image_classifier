import torch
from torchvision import transforms
from PIL import Image
import os
import sys
import random

# ✅ Add root to sys.path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import RealFakeCNN

def predict_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    try:
        image = Image.open(img_path).convert("RGB")
    except FileNotFoundError:
        print(f"[ERROR] File not found: {img_path}")
        return None

    image = transform(image).unsqueeze(0)  # Add batch dimension

    model = RealFakeCNN()
    model.load_state_dict(torch.load('model/real_fake_model.pth', map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        class_names = ['Fake', 'Real']
        return class_names[predicted.item()]

def get_random_image_from_folder(folder_path):
    try:
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files:
            print(f"[ERROR] No image files found in: {folder_path}")
            return None
        return os.path.join(folder_path, random.choice(files))
    except FileNotFoundError:
        print(f"[ERROR] Folder not found: {folder_path}")
        return None

if __name__ == "__main__":
    # Choose test class folder: 'real' or 'fake'
    test_folder = "data/test/real"  # You can also switch to 'real'

    img_path = get_random_image_from_folder(test_folder)

    if img_path:
        print(f"Testing on image: {img_path}")
        result = predict_image(img_path)
        if result:
            print(f"Prediction: {result}")
