from ultralytics import YOLO
import csv
import os


def main():
    # Selected model
    model_path = r"runs\detect\runs\experiments\yolov8n_768epoch50\weights\best.pt"

    # Dataset
    data_path = r"dataset_detection\data.yaml"

    # Validation confidence thresholds
    thresholds = [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60
    ]

    # Output directory
    output_dir = r"runs\evaluation\confidence_threshold"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("CONFIDENCE THRESHOLD ANALYSIS")
    print("=" * 70)
    print(f"Model : {model_path}")
    print(f"Data  : {data_path}")
    print("Split : validation")
    print("=" * 70)

    # Load model
    model = YOLO(model_path)

    results = []

    for conf in thresholds:
        print()
        print("-" * 70)
        print(f"Evaluating confidence threshold: {conf:.2f}")
        print("-" * 70)

        metrics = model.val(
            data=data_path,
            split="val",
            imgsz=768,
            batch=4,
            device=0,
            workers=0,
            conf=conf,
            plots=False,
            verbose=False
        )

        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)
        map50 = float(metrics.box.map50)
        map5095 = float(metrics.box.map)

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        results.append({
            "confidence": conf,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mAP50": map50,
            "mAP50-95": map5095
        })

        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1-score  : {f1:.4f}")
        print(f"mAP50     : {map50:.4f}")
        print(f"mAP50-95  : {map5095:.4f}")

    # Save results
    csv_path = os.path.join(
        output_dir,
        "confidence_threshold_results.csv"
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "confidence",
                "precision",
                "recall",
                "f1",
                "mAP50",
                "mAP50-95"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    # Find best F1 threshold
    best = max(results, key=lambda x: x["f1"])

    print()
    print("=" * 70)
    print("CONFIDENCE THRESHOLD ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Best validation F1 threshold: "
        f"{best['confidence']:.2f}"
    )

    print(f"Precision : {best['precision']:.4f}")
    print(f"Recall    : {best['recall']:.4f}")
    print(f"F1-score  : {best['f1']:.4f}")
    print(f"mAP50     : {best['mAP50']:.4f}")
    print(f"mAP50-95  : {best['mAP50-95']:.4f}")

    print()
    print(f"Results saved to:")
    print(csv_path)
    print("=" * 70)


if __name__ == "__main__":
    main()