from ultralytics import YOLO
import torch


def main():

    print("=" * 70)
    print("BONE FRACTURE DETECTION")
    print("YOLOv8s - 768px FULL EXPERIMENT")
    print("=" * 70)

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print()
    print("Loading pretrained YOLOv8s...")

    model = YOLO("yolov8s.pt")

    print("Model loaded.")
    print()
    print("Starting 768px YOLOv8s training...")
    print()

    model.train(
        data="dataset_detection/data.yaml",

        # Training
        epochs=50,
        imgsz=768,
        batch=4,

        # GPU
        device=0,

        # Windows
        workers=0,

        # Reproducibility
        seed=42,

        # Early stopping
        patience=15,

        # Output
        project="runs/experiments",
        name="yolov8s_768_epoch50",

        # Validation
        val=True,

        # Save
        save=True,
        plots=True
    )

    print()
    print("=" * 70)
    print("YOLOv8s 768px FULL EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()