import torch
from models.mobilenet_v2_tsm import MobileNetV2TSM

DEVICE = "cpu"

def main():
    model = MobileNetV2TSM(num_classes=27, num_segments=8, pretrained=False)
    model.load_state_dict(torch.load("checkpoints/mobilenetv2_tsm_epoch10.pth", map_location=DEVICE))
    model.eval()

    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    torch.save(quantized_model.state_dict(), "checkpoints/mobilenetv2_tsm_int8.pth")
    print("Quantized model saved.")

if __name__ == "__main__":
    main()