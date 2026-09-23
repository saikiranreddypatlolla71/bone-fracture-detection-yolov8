from ultralytics import YOLO
import torch


def main():

    print("=" * 70)
    print("BONE FRACTURE DETECTION")
    print("YOLOv8n - 768px X-RAY AUGMENTATION EXPERIMENT")
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
    print("Starting X-ray augmentation experiment...")
    print()

    model.train(
        data="dataset_detection/data.yaml",

        # Training
        epochs=50,
        imgsz=768,
        batch=4,

        # GPU
        device=0,
        workers=0,

        # Reproducibility
        seed=42,

        # Early stopping
        patience=15,

        # --------------------------------------------------
        # X-RAY-SPECIFIC AUGMENTATION
        # --------------------------------------------------

        # Disable color augmentation
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,

        # Mild geometric augmentation
        degrees=3.0,
        translate=0.05,
        scale=0.20,
        shear=0.0,
        perspective=0.0,

        # Avoid left/right flipping because X-rays
        # can contain laterality markers.
        fliplr=0.0,
        flipud=0.0,

        # Disable synthetic image mixing
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,

        # Keep normal image erasing disabled
        erasing=0.0,

        # Output
        project="runs/experiments",
        name="yolov8n_768_xray_aug",

        val=True,
        save=True,
        plots=True
    )

    print()
    print("=" * 70)
    print("X-RAY AUGMENTATION EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()