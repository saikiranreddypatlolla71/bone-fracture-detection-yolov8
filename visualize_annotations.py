from pathlib import Path
import random
import cv2
import math

DATASET = Path("dataset_detection")
OUTPUT = Path("annotation_preview")

random.seed(42)

CLASS_NAMES = [
    "elbow positive",
    "fingers positive",
    "forearm fracture",
    "humerus fracture",
    "humerus",
    "shoulder fracture",
    "wrist positive"
]

SAMPLES_PER_SPLIT = 10

OUTPUT.mkdir(exist_ok=True)

for split in ["train", "valid", "test"]:

    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    images = list(image_dir.glob("*"))

    random.shuffle(images)
    selected = images[:SAMPLES_PER_SPLIT]

    for image_path in selected:

        label_path = label_dir / (image_path.stem + ".txt")

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        h, w = image.shape[:2]

        if label_path.exists():

            for line in label_path.read_text().splitlines():

                parts = line.split()

                if len(parts) != 5:
                    continue

                cls, xc, yc, bw, bh = map(float, parts)

                cls = int(cls)

                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)

                # Keep coordinates inside image
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = CLASS_NAMES[cls]

                cv2.putText(
                    image,
                    label,
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

        # Add split name
        cv2.putText(
            image,
            split.upper(),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        output_path = OUTPUT / f"{split}_{image_path.stem}.jpg"

        cv2.imwrite(str(output_path), image)

print("=" * 70)
print("ANNOTATION VISUALIZATION COMPLETE")
print("=" * 70)

for split in ["train", "valid", "test"]:
    count = len(list(OUTPUT.glob(f"{split}_*.jpg")))
    print(f"{split}: {count} preview images")

print()
print(f"Preview folder:")
print(OUTPUT.resolve())