import csv
from pathlib import Path


def main():

    output_dir = Path(
        r"runs\evaluation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Experiment results
    # ---------------------------------------------------------

    experiments = [

        {
            "experiment": "Baseline YOLOv8n",
            "model": "YOLOv8n",
            "imgsz": 640,
            "epochs": 10,
            "batch": 8,
            "val_precision": 0.104,
            "val_recall": 0.183,
            "val_map50": 0.139,
            "val_map50_95": 0.0564,
            "test_precision": "",
            "test_recall": "",
            "test_map50": "",
            "test_map50_95": "",
            "notes": "Initial baseline experiment"
        },

        {
            "experiment": "YOLOv8n 640x640",
            "model": "YOLOv8n",
            "imgsz": 640,
            "epochs": 50,
            "batch": 8,
            "val_precision": 0.189,
            "val_recall": 0.302,
            "val_map50": 0.209,
            "val_map50_95": 0.0922,
            "test_precision": "",
            "test_recall": "",
            "test_map50": "",
            "test_map50_95": "",
            "notes": "50-epoch experiment; early stopped"
        },

        {
            "experiment": "YOLOv8n 768x768",
            "model": "YOLOv8n",
            "imgsz": 768,
            "epochs": 50,
            "batch": 4,
            "val_precision": 0.261,
            "val_recall": 0.280,
            "val_map50": 0.245,
            "val_map50_95": 0.106,
            "test_precision": 0.226,
            "test_recall": 0.320,
            "test_map50": 0.205,
            "test_map50_95": 0.0834,
            "notes": "Selected model; independent test evaluation"
        },

        {
            "experiment": "YOLOv8n 768 X-ray augmentation",
            "model": "YOLOv8n",
            "imgsz": 768,
            "epochs": 50,
            "batch": 4,
            "val_precision": 0.211,
            "val_recall": 0.177,
            "val_map50": 0.138,
            "val_map50_95": 0.0507,
            "test_precision": "",
            "test_recall": "",
            "test_map50": "",
            "test_map50_95": "",
            "notes": "Custom X-ray-specific augmentation experiment"
        },

        {
            "experiment": "YOLOv8s 768x768",
            "model": "YOLOv8s",
            "imgsz": 768,
            "epochs": 50,
            "batch": 4,
            "val_precision": 0.2212,
            "val_recall": 0.2558,
            "val_map50": 0.2188,
            "val_map50_95": 0.0903,
            "test_precision": "",
            "test_recall": "",
            "test_map50": "",
            "test_map50_95": "",
            "notes": "Larger model comparison"
        },

        {
            "experiment": "Selected model - threshold 0.20",
            "model": "YOLOv8n",
            "imgsz": 768,
            "epochs": 50,
            "batch": 4,
            "val_precision": 0.3104,
            "val_recall": 0.2333,
            "val_map50": 0.1977,
            "val_map50_95": 0.0881,
            "test_precision": 0.246,
            "test_recall": 0.240,
            "test_map50": 0.152,
            "test_map50_95": 0.0616,
            "notes": "Confidence threshold locked at 0.20 using validation F1"
        }
    ]

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    csv_path = (
        output_dir /
        "final_experiment_summary.csv"
    )

    fieldnames = [
        "experiment",
        "model",
        "imgsz",
        "epochs",
        "batch",
        "val_precision",
        "val_recall",
        "val_map50",
        "val_map50_95",
        "test_precision",
        "test_recall",
        "test_map50",
        "test_map50_95",
        "notes"
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(experiments)

    # ---------------------------------------------------------
    # Create Markdown report
    # ---------------------------------------------------------

    md_path = (
        output_dir /
        "FINAL_EXPERIMENT_SUMMARY.md"
    )

    lines = []

    lines.append("# Bone Fracture Detection - Final Experiment Summary")
    lines.append("")
    lines.append("## Selected model")
    lines.append("")
    lines.append(
        "YOLOv8n at 768x768 pixels was selected based on "
        "the completed validation experiments."
    )
    lines.append("")
    lines.append("## Experiment comparison")
    lines.append("")
    lines.append(
        "| Experiment | Model | Image Size | Epochs | "
        "Val Precision | Val Recall | Val mAP50 | Val mAP50-95 |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|---:|"
    )

    for exp in experiments:

        lines.append(
            f"| {exp['experiment']} "
            f"| {exp['model']} "
            f"| {exp['imgsz']} "
            f"| {exp['epochs']} "
            f"| {exp['val_precision']} "
            f"| {exp['val_recall']} "
            f"| {exp['val_map50']} "
            f"| {exp['val_map50_95']} |"
        )

    lines.append("")
    lines.append("## Independent test evaluation")
    lines.append("")
    lines.append(
        "The selected YOLOv8n 768x768 model was evaluated "
        "on the held-out test set of 388 images."
    )
    lines.append("")
    lines.append(
        "- Standard test evaluation:"
    )
    lines.append(
        "  - Precision: 0.226"
    )
    lines.append(
        "  - Recall: 0.320"
    )
    lines.append(
        "  - mAP50: 0.205"
    )
    lines.append(
        "  - mAP50-95: 0.0834"
    )
    lines.append("")
    lines.append(
        "- Locked confidence threshold 0.20:"
    )
    lines.append(
        "  - Precision: 0.246"
    )
    lines.append(
        "  - Recall: 0.240"
    )
    lines.append(
        "  - mAP50: 0.152"
    )
    lines.append(
        "  - mAP50-95: 0.0616"
    )

    lines.append("")
    lines.append("## Threshold analysis")
    lines.append("")
    lines.append(
        "The validation confidence sweep identified 0.20 "
        "as the maximum-F1 operating threshold."
    )
    lines.append("")
    lines.append(
        "- Confidence threshold: 0.20"
    )
    lines.append(
        "- Validation precision: 0.3104"
    )
    lines.append(
        "- Validation recall: 0.2333"
    )
    lines.append(
        "- Validation F1: 0.2664"
    )

    lines.append("")
    lines.append("## Error analysis")
    lines.append("")
    lines.append(
        "At confidence 0.20 and IoU matching threshold 0.50, "
        "the custom test-set error analysis produced:"
    )
    lines.append("")
    lines.append(
        "- True positives: 51"
    )
    lines.append(
        "- False positives: 140"
    )
    lines.append(
        "- False negatives: 173"
    )
    lines.append(
        "- Custom precision: 0.2670"
    )
    lines.append(
        "- Custom recall: 0.2277"
    )
    lines.append(
        "- Custom F1: 0.2458"
    )

    lines.append("")
    lines.append("## Inference performance")
    lines.append("")
    lines.append(
        "- Hardware: NVIDIA GeForce RTX 3050 6GB"
    )
    lines.append(
        "- Input resolution: 768x768"
    )
    lines.append(
        "- Test images: 388"
    )
    lines.append(
        "- Average benchmark run time: 3.820 seconds"
    )
    lines.append(
        "- Latency per image: 9.85 ms"
    )
    lines.append(
        "- Throughput: 101.57 images/second"
    )

    lines.append("")
    lines.append("## Important dataset limitation")
    lines.append("")
    lines.append(
        "The humerus fracture class contains only three "
        "annotated objects in the complete dataset. After "
        "source-level grouping and splitting, all three "
        "were assigned to the training partition, leaving "
        "no validation or test instances for independent "
        "evaluation of this class."
    )

    lines.append("")
    lines.append("## Interpretation note")
    lines.append("")
    lines.append(
        "The reported results evaluate agreement between "
        "model predictions and dataset annotations. "
        "They should not be interpreted as clinical diagnosis "
        "or as evidence of clinical diagnostic performance."
    )

    with open(
        md_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("\n".join(lines))

    # ---------------------------------------------------------
    # Print completion
    # ---------------------------------------------------------

    print("=" * 70)
    print("FINAL EXPERIMENT SUMMARY CREATED")
    print("=" * 70)

    print("CSV:")
    print(csv_path)

    print()
    print("Markdown report:")
    print(md_path)

    print()
    print("Selected model:")
    print("YOLOv8n 768x768")

    print()
    print("Standard independent test:")
    print("Precision : 0.226")
    print("Recall    : 0.320")
    print("mAP50     : 0.205")
    print("mAP50-95  : 0.0834")

    print("=" * 70)


if __name__ == "__main__":
    main()