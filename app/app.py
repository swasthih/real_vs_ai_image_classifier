import os
import sys
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from torchvision import transforms

# Import model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import RealFakeCNN

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
model = RealFakeCNN()
model.load_state_dict(torch.load("model/real_fake_model.pth", map_location=torch.device("cpu")))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32), interpolation=Image.LANCZOS),  # Better quality resize
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_url = None
    error = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename.strip() == "":
            error = "Please select an image file."
        else:
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            try:
                # Open image and convert to RGB
                image = Image.open(save_path).convert("RGB")
                image_tensor = transform(image).unsqueeze(0)

                # Predict
                with torch.no_grad():
                    output = model(image_tensor)
                    _, pred = torch.max(output, 1)
                    prediction = "REAL" if pred.item() == 1 else "FAKE"

                image_url = f"uploads/{filename}"
            except Exception as e:
                error = f"Error processing image: {str(e)}"

    return render_template("index.html",
                           prediction=prediction,
                           image_url=image_url,
                           error=error)

if __name__ == "__main__":
    app.run(debug=True)
