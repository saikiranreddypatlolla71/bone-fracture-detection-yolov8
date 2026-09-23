from ultralytics import YOLO
from pathlib import Path
import random
import cv2


def main():
    model_path = Path(
        r"runs\detect\runs\experiments\yolov8n_768epoch50\weights\best.pt"
    )

    image_dir = Path(r"dataset_detection\test\images")
    label_dir = Path(r"dataset_detection\test\labels")

    output_dir = Path(
        r"runs\evaluation\explainability"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = [
        "elbow positive",
        "fingers positive",
        "forearm fracture",
        "humerus fracture",
        "humerus",
        "shoulder fracture",
        "wrist positive",
    ]

    print("=" * 70)
    print("EXPLAINABILITY / ERROR VISUALIZATION")
    print("=" * 70)

    model = YOLO(str(model_path))

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    image_paths = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ]

    image_paths.sort()

    # Fixed seed for reproducibility
    random.seed(42)

    # Select up to 12 test images
    selected_images = random.sample(
        image_paths,
        min(12, len(image_paths))
    )

    print(f"Total test images : {len(image_paths)}")
    print(f"Selected images   : {len(selected_images)}")
    print()

    for index, image_path in enumerate(selected_images, start=1):

        print(
            f"[{index}/{len(selected_images)}] "
            f"Processing {image_path.name}"
        )

        image = cv2.imread(str(image_path))

        if image is None:
            print("  WARNING: Could not read image.")
            continue

        # ---------------------------------------------------------
        # Ground-truth boxes
        # ---------------------------------------------------------

        label_path = label_dir / f"{image_path.stem}.txt"

        ground_truth = []

        if label_path.exists():

            h, w = image.shape[:2]

            with open(
                label_path,
                "r",
                encoding="utf-8"
            ) as f:

                for line in f:

                    parts = line.strip().split()

                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])

                    x_center = float(parts[1]) * w
                    y_center = float(parts[2]) * h
                    box_width = float(parts[3]) * w
                    box_height = float(parts[4]) * h

                    x1 = int(x_center - box_width / 2)
                    y1 = int(y_center - box_height / 2)
                    x2 = int(x_center + box_width / 2)
                    y2 = int(y_center + box_height / 2)

                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w - 1))
                    y2 = max(0, min(y2, h - 1))

                    ground_truth.append(
                        (
                            class_id,
                            x1,
                            y1,
                            x2,
                            y2
                        )
                    )

        # ---------------------------------------------------------
        # Model predictions
        # ---------------------------------------------------------

        results = model.predict(
            source=str(image_path),
            imgsz=768,
            conf=0.20,
            device=0,
            verbose=False
        )

        result = results[0]

        prediction_image = image.copy()

        if result.boxes is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for box, cls, confidence in zip(
                boxes,
                classes,
                confidences
            ):

                x1, y1, x2, y2 = map(int, box)

                class_id = int(cls)

                if 0 <= class_id < len(class_names):
                    class_name = class_names[class_id]
                else:
                    class_name = f"class_{class_id}"

                label = (
                    f"{class_name} "
                    f"{confidence:.2f}"
                )

                cv2.rectangle(
                    prediction_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    prediction_image,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

        # ---------------------------------------------------------
        # Combined image: ground truth + predictions
        # ---------------------------------------------------------

        combined_image = image.copy()

        # Ground truth = blue
        for class_id, x1, y1, x2, y2 in ground_truth:

            if 0 <= class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = f"class_{class_id}"

            label = f"GT: {class_name}"

            cv2.rectangle(
                combined_image,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            cv2.putText(
                combined_image,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 0, 0),
                2,
                cv2.LINE_AA
            )

        # Predictions = green
        if result.boxes is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for box, cls, confidence in zip(
                boxes,
                classes,
                confidences
            ):

                x1, y1, x2, y2 = map(int, box)

                class_id = int(cls)

                if 0 <= class_id < len(class_names):
                    class_name = class_names[class_id]
                else:
                    class_name = f"class_{class_id}"

                label = (
                    f"Pred: {class_name} "
                    f"{confidence:.2f}"
                )

                cv2.rectangle(
                    combined_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    combined_image,
                    label,
                    (x1, min(
                        combined_image.shape[0] - 10,
                        y2 + 20
                    )),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

        # ---------------------------------------------------------
        # Save outputs
        # ---------------------------------------------------------

        output_name = f"{index:02d}_{image_path.stem}"

        cv2.imwrite(
            str(output_dir / f"{output_name}_prediction.jpg"),
            prediction_image
        )

        cv2.imwrite(
            str(output_dir / f"{output_name}_comparison.jpg"),
            combined_image
        )

    print()
    print("=" * 70)
    print("EXPLAINABILITY VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"Output folder:")
    print(output_dir)
    print()
    print("Files created:")
    print("  *_prediction.jpg")
    print("  *_comparison.jpg")
    print("=" * 70)


if __name__ == "__main__":
    main()