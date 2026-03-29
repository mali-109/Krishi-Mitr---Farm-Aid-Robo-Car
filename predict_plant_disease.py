import cv2
import torch
from torchvision import transforms, models
from PIL import Image

# -------------------------------
# 1. Load the model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3  # Marigold model has 3 classes

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("/home/satyam/plant_disease_model.pth", map_location=device))
model = model.to(device)
model.eval()

# -------------------------------
# 2. Define classes
# -------------------------------
classes = ['Tagetes_erecta_alternaria', 'Leaf_blight', 'Healthy_leaf']  # Replace with your 3 classes

# -------------------------------
# 3. Image preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------
# 4. Open camera and predict
# -------------------------------
cap = cv2.VideoCapture(0)  # 0 = default camera

print("Press SPACE to capture image for disease detection, ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Camera - Press SPACE to Capture", frame)
    key = cv2.waitKey(1)

    if key % 256 == 27:  # ESC pressed
        print("Exiting...")
        break
    elif key % 256 == 32:  # SPACE pressed
        # Convert OpenCV frame (BGR) to PIL Image (RGB)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)

        # Transform and add batch dimension
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        print(f"Prediction: {classes[pred.item()]}, Confidence: {conf.item()*100:.2f}%")

cap.release()
cv2.destroyAllWindows()

