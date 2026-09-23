from ultralytics import YOLO
from pathlib import Path
import cv2
import csv


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes.

    Box format:
    [x1, y1, x2, y2]
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection_area = (
        intersection_width * intersection_height
    )

    area1 = max(0, box1[2] - box1[0]) * max(
        0, box1[3] - box1[1]
    )

    area2 = max(0, box2[2] - box2[0]) * max(
        0, box2[3] - box2[1]
    )

    union_area = area1 + area2 - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def load_ground_truth(label_path, image_width, image_height):
    """
    Load YOLO-format ground-truth boxes.
    """

    ground_truth = []

    if not label_path.exists():
        return ground_truth

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

            x_center = float(parts[1]) * image_width
            y_center = float(parts[2]) * image_height
            box_width = float(parts[3]) * image_width
            box_height = float(parts[4]) * image_height

            x1 = x_center - box_width / 2
            y1 = y_center - box_height / 2
            x2 = x_center + box_width / 2
            y2 = y_center + box_height / 2

            ground_truth.append(
                {
                    "class_id": class_id,
                    "box": [x1, y1, x2, y2],
                }
            )

    return ground_truth


def main():

    model_path = Path(
        r"runs\detect\runs\experiments\yolov8n_768epoch50\weights\best.pt"
    )

    image_dir = Path(
        r"dataset_detection\test\images"
    )

    label_dir = Path(
        r"dataset_detection\test\labels"
    )

    output_dir = Path(
        r"runs\evaluation\error_analysis"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    confidence_threshold = 0.20
    iou_threshold = 0.50

    class_names = [
        "elbow positive",
        "fingers positive",
        "forearm fracture",
        "humerus fracture",
        "humerus",
        "shoulder fracture",
        "wrist positive",
    ]

    number_of_classes = len(class_names)

    # Per-class counters
    tp = [0] * number_of_classes
    fp = [0] * number_of_classes
    fn = [0] * number_of_classes

    total_images = 0
    total_predictions = 0
    total_ground_truth = 0

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    image_paths = sorted(
        [
            p
            for p in image_dir.iterdir()
            if p.suffix.lower() in image_extensions
        ]
    )

    print("=" * 70)
    print("TEST-SET AUTOMATIC ERROR ANALYSIS")
    print("=" * 70)
    print(f"Model               : {model_path}")
    print(f"Test images         : {len(image_paths)}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"IoU threshold       : {iou_threshold}")
    print("=" * 70)

    model = YOLO(str(model_path))

    # Process each image individually
    for image_number, image_path in enumerate(
        image_paths,
        start=1
    ):

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        height, width = image.shape[:2]

        label_path = label_dir / f"{image_path.stem}.txt"

        ground_truth = load_ground_truth(
            label_path,
            width,
            height
        )

        # Run model
        results = model.predict(
            source=str(image_path),
            imgsz=768,
            conf=confidence_threshold,
            device=0,
            verbose=False
        )

        result = results[0]

        predictions = []

        if result.boxes is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for box, class_id, confidence in zip(
                boxes,
                classes,
                confidences
            ):

                predictions.append(
                    {
                        "class_id": int(class_id),
                        "box": box.tolist(),
                        "confidence": float(confidence),
                    }
                )

        total_images += 1
        total_predictions += len(predictions)
        total_ground_truth += len(ground_truth)

        # Track matched GT and predictions
        matched_gt = set()
        matched_predictions = set()

        # Create possible matches
        candidates = []

        for pred_index, prediction in enumerate(
            predictions
        ):

            for gt_index, gt in enumerate(
                ground_truth
            ):

                # Class must match
                if prediction["class_id"] != gt["class_id"]:
                    continue

                iou = calculate_iou(
                    prediction["box"],
                    gt["box"]
                )

                if iou >= iou_threshold:

                    candidates.append(
                        (
                            iou,
                            pred_index,
                            gt_index
                        )
                    )

        # Highest-IoU matches first
        candidates.sort(
            reverse=True,
            key=lambda x: x[0]
        )

        for iou, pred_index, gt_index in candidates:

            if pred_index in matched_predictions:
                continue

            if gt_index in matched_gt:
                continue

            class_id = predictions[pred_index]["class_id"]

            tp[class_id] += 1

            matched_predictions.add(pred_index)
            matched_gt.add(gt_index)

        # Unmatched predictions = false positives
        for pred_index, prediction in enumerate(
            predictions
        ):

            if pred_index not in matched_predictions:

                class_id = prediction["class_id"]

                if 0 <= class_id < number_of_classes:
                    fp[class_id] += 1

        # Unmatched ground truths = false negatives
        for gt_index, gt in enumerate(
            ground_truth
        ):

            if gt_index not in matched_gt:

                class_id = gt["class_id"]

                if 0 <= class_id < number_of_classes:
                    fn[class_id] += 1

        if image_number % 25 == 0:
            print(
                f"Processed "
                f"{image_number}/{len(image_paths)} images"
            )

    # -------------------------------------------------------------
    # Calculate metrics
    # -------------------------------------------------------------

    total_tp = sum(tp)
    total_fp = sum(fp)
    total_fn = sum(fn)

    if total_tp + total_fp > 0:
        overall_precision = (
            total_tp /
            (total_tp + total_fp)
        )
    else:
        overall_precision = 0.0

    if total_tp + total_fn > 0:
        overall_recall = (
            total_tp /
            (total_tp + total_fn)
        )
    else:
        overall_recall = 0.0

    if overall_precision + overall_recall > 0:
        overall_f1 = (
            2 *
            overall_precision *
            overall_recall /
            (overall_precision + overall_recall)
        )
    else:
        overall_f1 = 0.0

    # -------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("OVERALL ERROR ANALYSIS")
    print("=" * 70)

    print(f"Images processed : {total_images}")
    print(f"Ground-truth     : {total_ground_truth}")
    print(f"Predictions      : {total_predictions}")
    print()
    print(f"True positives   : {total_tp}")
    print(f"False positives  : {total_fp}")
    print(f"False negatives  : {total_fn}")
    print()
    print(f"Precision        : {overall_precision:.4f}")
    print(f"Recall           : {overall_recall:.4f}")
    print(f"F1-score         : {overall_f1:.4f}")

    print()
    print("=" * 70)
    print("PER-CLASS ERROR ANALYSIS")
    print("=" * 70)

    print(
        f"{'Class':25s}"
        f"{'TP':>8s}"
        f"{'FP':>8s}"
        f"{'FN':>8s}"
        f"{'Precision':>12s}"
        f"{'Recall':>10s}"
    )

    csv_rows = []

    for class_id, class_name in enumerate(
        class_names
    ):

        class_tp = tp[class_id]
        class_fp = fp[class_id]
        class_fn = fn[class_id]

        if class_tp + class_fp > 0:
            class_precision = (
                class_tp /
                (class_tp + class_fp)
            )
        else:
            class_precision = 0.0

        if class_tp + class_fn > 0:
            class_recall = (
                class_tp /
                (class_tp + class_fn)
            )
        else:
            class_recall = 0.0

        print(
            f"{class_name:25s}"
            f"{class_tp:8d}"
            f"{class_fp:8d}"
            f"{class_fn:8d}"
            f"{class_precision:12.4f}"
            f"{class_recall:10.4f}"
        )

        csv_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "TP": class_tp,
                "FP": class_fp,
                "FN": class_fn,
                "precision": class_precision,
                "recall": class_recall,
            }
        )

    # -------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------

    csv_path = (
        output_dir /
        "test_error_analysis.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_id",
                "class_name",
                "TP",
                "FP",
                "FN",
                "precision",
                "recall",
            ]
        )

        writer.writeheader()
        writer.writerows(csv_rows)

    print()
    print("=" * 70)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"CSV saved to:")
    print(csv_path)
    print("=" * 70)


if __name__ == "__main__":
    main()