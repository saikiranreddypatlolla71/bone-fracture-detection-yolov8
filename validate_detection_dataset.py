from pathlib import Path
from PIL import Image

DATASET = Path(
    r"C:\Users\saiki\bone-fracture-detection\dataset_detection"
)

CLASS_COUNT = 7

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

errors = []

total_images = 0
total_labels = 0
total_objects = 0

for split in ["train", "valid", "test"]:

    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    images = [
        p for p in image_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = list(label_dir.glob("*.txt"))

    total_images += len(images)
    total_labels += len(labels)

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    for stem in image_stems - label_stems:
        errors.append(
            f"{split}: missing label for {stem}"
        )

    for stem in label_stems - image_stems:
        errors.append(
            f"{split}: missing image for {stem}"
        )

    for label in labels:

        try:
            lines = label.read_text(
                encoding="utf-8"
            ).splitlines()
        except Exception as e:
            errors.append(
                f"{label}: cannot read label: {e}"
            )
            continue

        for line_no, line in enumerate(lines, 1):

            if not line.strip():
                continue

            values = line.split()

            # Must be exactly:
            # class x y width height

            if len(values) != 5:

                errors.append(
                    f"{label}:{line_no}: "
                    f"expected 5 values, got {len(values)}"
                )
                continue

            try:

                class_id = int(values[0])

                x = float(values[1])
                y = float(values[2])
                w = float(values[3])
                h = float(values[4])

            except ValueError:

                errors.append(
                    f"{label}:{line_no}: "
                    "non-numeric value"
                )
                continue

            total_objects += 1

            if not 0 <= class_id < CLASS_COUNT:

                errors.append(
                    f"{label}:{line_no}: "
                    f"invalid class {class_id}"
                )

            if not 0 <= x <= 1:
                errors.append(
                    f"{label}:{line_no}: "
                    f"x={x} outside 0-1"
                )

            if not 0 <= y <= 1:
                errors.append(
                    f"{label}:{line_no}: "
                    f"y={y} outside 0-1"
                )

            if not 0 < w <= 1:
                errors.append(
                    f"{label}:{line_no}: "
                    f"width={w} invalid"
                )

            if not 0 < h <= 1:
                errors.append(
                    f"{label}:{line_no}: "
                    f"height={h} invalid"
                )

            # Bounding box must stay inside image
            if x - w / 2 < 0:
                errors.append(
                    f"{label}:{line_no}: "
                    "bbox extends left of image"
                )

            if x + w / 2 > 1:
                errors.append(
                    f"{label}:{line_no}: "
                    "bbox extends right of image"
                )

            if y - h / 2 < 0:
                errors.append(
                    f"{label}:{line_no}: "
                    "bbox extends above image"
                )

            if y + h / 2 > 1:
                errors.append(
                    f"{label}:{line_no}: "
                    "bbox extends below image"
                )


print("=" * 70)
print("YOLO DETECTION DATASET VALIDATION")
print("=" * 70)

print(f"Images       : {total_images}")
print(f"Label files  : {total_labels}")
print(f"Objects      : {total_objects}")
print(f"Errors       : {len(errors)}")

if errors:

    print("\nFIRST 50 ERRORS")
    print("-" * 70)

    for error in errors[:50]:
        print(error)

else:

    print("\nALL CHECKS PASSED")