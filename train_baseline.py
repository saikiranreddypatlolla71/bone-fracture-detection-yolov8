from ultralytics import YOLO
import torch


def main():

    print("=" * 70)
    print("BONE FRACTURE DETECTION - YOLOv8 BASELINE")
    print("=" * 70)

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print()
    print("Loading YOLOv8n model...")

    model = YOLO("yolov8n.pt")

    print("Model loaded successfully.")
    print()
    print("Starting baseline training...")
    print()

    model.train(
        data="dataset_detection/data.yaml",

        # Training
        epochs=10,
        imgsz=640,
        batch=8,

        # GPU
        device=0,

        # Windows-safe DataLoader
        workers=0,

        # Reproducibility
        seed=42,

        # Output
        project="runs/baseline",
        name="yolov8n_10epoch",

        # Validation and plots
        val=True,
        save=True,
        plots=True
    )

    print()
    print("=" * 70)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()