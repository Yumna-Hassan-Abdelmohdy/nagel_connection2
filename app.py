from flask import Flask, request, jsonify
import torch
import timm
from torchvision import transforms
from PIL import Image
import io
import os
import requests

app = Flask(__name__)

# إعدادات النموذج
model_path = "Nagel_model.pth"
model_url = "https://drive.google.com/uc?export=download&id=10rpMMep8aYMXI8HE3lUvJucNrWQ7xXkl" # عدّل الرابط هنا
num_classes = 10
class_names = [
    'Acral Lentiginous Melanoma', 'Beaus Line', 'Blue Finger', 'Clubbing',
    'Error-Not Nail', 'Healthy Nail', 'Koilonychia', 'Muehrckes Lines',
    'Pitting', 'Terrys Nail'
]

# تحميل الموديل من الإنترنت لو مش موجود
if not os.path.exists(model_path):
    print("📥 Downloading model weights...")
    r = requests.get(model_url)
    with open(model_path, 'wb') as f:
        f.write(r.content)
    print("✅ Model downloaded.")

# تحميل الموديل
device = torch.device("cpu")
model = timm.create_model("resnet18d", pretrained=False, num_classes=num_classes)

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model loaded and ready!")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

# تجهيز الصورة
preprocess = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(image: Image.Image):
    image = image.convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
    pred_index = probs.argmax().item()
    return {
        "class": class_names[pred_index],
        "confidence": round(probs[pred_index].item() * 100, 2),
        "probabilities": {class_names[i]: round(probs[i].item() * 100, 2) for i in range(len(class_names))}
    }

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🧠 Nail Disease Classifier is running!"})

@app.route("/predict/", methods=["POST"])
def classify():
    if 'file' not in request.files:
        return jsonify({"error": "❌ No image uploaded."}), 400
    try:
        image = Image.open(io.BytesIO(request.files['file'].read()))
    except Exception as e:
        return jsonify({"error": f"❌ Failed to open image: {e}"}), 400
    result = predict(image)
    return jsonify(result)

if __name__ == "__main__":
    app.run()
