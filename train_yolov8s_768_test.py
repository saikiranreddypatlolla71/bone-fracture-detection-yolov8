from ultralytics import YOLO
import torch


def main():

    print("=" * 70)
    print("BONE FRACTURE DETECTION")
    print("YOLOv8s - 768px HARDWARE TEST")
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
    print("Starting 3-epoch test...")

    model.train(
        data="dataset_detection/data.yaml",

        epochs=3,
        imgsz=768,
        batch=4,

        device=0,
        workers=0,

        seed=42,

        project="runs/experiments",
        name="yolov8s_768_test",

        val=True,
        save=True,
        plots=True
    )

    print()
    print("=" * 70)
    print("YOLOv8s 768px TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()