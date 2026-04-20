import time
import torch
import psutil
import os
from models.mobilenet_v2_tsm import MobileNetV2TSM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def benchmark():
    model = MobileNetV2TSM(num_classes=27, num_segments=8, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/mobilenetv2_tsm_epoch10.pth", map_location=DEVICE))
    model.eval()

    dummy_input = torch.randn(1, 8, 3, 112, 112).to(DEVICE)

    for _ in range(10):
        _ = model(dummy_input)

    runs = 50
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)
    end = time.time()

    avg_latency = (end - start) / runs
    fps = 1.0 / avg_latency
    memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

    print(f"Average Latency: {avg_latency*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"Memory Usage: {memory_mb:.2f} MB")

if __name__ == "__main__":
    benchmark()