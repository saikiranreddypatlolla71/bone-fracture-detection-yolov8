from ultralytics import YOLO
import torch


def main():

    print("=" * 70)
    print("BONE FRACTURE DETECTION")
    print("YOLOv8n - 768px RESOLUTION TEST")
    print("=" * 70)

    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print()
    print("Loading YOLOv8n...")

    model = YOLO("yolov8n.pt")

    print("Model loaded.")
    print()
    print("Starting 3-epoch 768px test...")

    model.train(
        data="dataset_detection/data.yaml",

        # Short hardware test
        epochs=3,

        # Higher X-ray resolution
        imgsz=768,

        # Reduce batch to control VRAM
        batch=4,

        # GPU
        device=0,

        # Windows-safe
        workers=0,

        # Reproducibility
        seed=42,

        # Output
        project="runs/experiments",
        name="yolov8n_768_test",

        # Validation
        val=True,

        save=True,
        plots=True
    )

    print()
    print("=" * 70)
    print("768px TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()