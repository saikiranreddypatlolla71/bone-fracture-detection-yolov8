from ultralytics import YOLO
import torch


def main():

    print("=" * 70)
    print("BONE FRACTURE DETECTION")
    print("YOLOv8n - 50 EPOCH EXPERIMENT")
    print("=" * 70)

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print()
    print("Loading pretrained YOLOv8n...")

    model = YOLO("yolov8n.pt")

    print("Model loaded.")
    print()
    print("Starting 50-epoch training...")
    print()

    model.train(
        data="dataset_detection/data.yaml",

        # Training
        epochs=50,
        imgsz=640,
        batch=8,

        # GPU
        device=0,

        # Windows
        workers=0,

        # Reproducibility
        seed=42,

        # Training behavior
        patience=15,

        # Output
        project="runs/experiments",
        name="yolov8n_50epoch",

        # Validation
        val=True,

        # Save results
        save=True,
        plots=True
    )

    print()
    print("=" * 70)
    print("50-EPOCH EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()