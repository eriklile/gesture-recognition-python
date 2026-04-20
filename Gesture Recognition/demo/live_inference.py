import cv2
import torch
import numpy as np
from torchvision import transforms
from collections import deque
from models.mobilenet_v2_tsm import MobileNetV2TSM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SEGMENTS = 8
IMG_SIZE = 112

CLASS_NAMES = ["Open Palm", "Fist", "Thumbs Up", "Peace", "Stop"]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

model = MobileNetV2TSM(
    num_classes=5, num_segments=NUM_SEGMENTS, pretrained=False
).to(DEVICE)

# Disable checkpoint loading for now
# model.load_state_dict(torch.load("checkpoints/mobilenetv2_tsm_epoch10.pth", map_location=DEVICE))

print("Warning: no trained checkpoint loaded. Using untrained model.")

model.eval()

buffer = deque(maxlen=NUM_SEGMENTS)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera 0 failed, trying camera 1...")
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Camera 1 failed, trying camera 2...")
    cap = cv2.VideoCapture(2)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open camera. Check macOS camera permissions, close other apps using the camera, and try a different camera index."
    )

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb)
    buffer.append(tensor)

    pred_text = "Collecting frames..."

    if len(buffer) == NUM_SEGMENTS:
        clip = torch.stack(list(buffer), dim=0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(clip)
            pred = output.argmax(dim=1).item()
            pred_text = CLASS_NAMES[pred]

    cv2.putText(display, f"Prediction: {pred_text}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Live Gesture Recognition", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
