from pathlib import Path
from PIL import Image
from collections import Counter
import hashlib

# ============================================================
# BONE FRACTURE DATASET AUDIT
# ============================================================

DATASET = Path(r"C:\Users\saiki\Downloads\archive (2)\BoneFractureYolo8")

SPLITS = ["train", "valid", "test"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_NAMES = {
    0: "elbow positive",
    1: "fingers positive",
    2: "forearm fracture",
    3: "humerus fracture",
    4: "humerus",
    5: "shoulder fracture",
    6: "wrist positive",
}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def get_images(folder):
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------
# Counters
# ------------------------------------------------------------

total_images = 0
total_labels = 0
total_annotations = 0

missing_labels = []
missing_images = []

corrupt_images = []
invalid_annotations = []
out_of_range = []

class_counter = Counter()
split_class_counter = {}

dimension_counter = Counter()
duplicate_hashes = {}

polygon_point_counter = Counter()

# ============================================================
# SPLIT AUDIT
# ============================================================

print("=" * 70)
print("BONE FRACTURE DATASET AUDIT")
print("=" * 70)
print()

for split in SPLITS:

    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    print(f"\n{'=' * 20} {split.upper()} {'=' * 20}")

    images = get_images(image_dir)
    labels = list(label_dir.glob("*.txt"))

    total_images += len(images)
    total_labels += len(labels)

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_label_stems = sorted(image_stems - label_stems)
    missing_image_stems = sorted(label_stems - image_stems)

    missing_labels.extend(
        [(split, x) for x in missing_label_stems]
    )

    missing_images.extend(
        [(split, x) for x in missing_image_stems]
    )

    print(f"Images : {len(images)}")
    print(f"Labels : {len(labels)}")

    print(f"Missing labels : {len(missing_label_stems)}")
    print(f"Missing images : {len(missing_image_stems)}")

    split_class_counter[split] = Counter()

    # --------------------------------------------------------
    # Image inspection
    # --------------------------------------------------------

    for img_path in images:

        try:
            with Image.open(img_path) as img:
                img.verify()

            with Image.open(img_path) as img:
                width, height = img.size

            dimension_counter[(width, height)] += 1

        except Exception as e:
            corrupt_images.append(
                (str(img_path), str(e))
            )

        # Duplicate detection
        try:
            h = file_hash(img_path)
            duplicate_hashes.setdefault(h, []).append(str(img_path))
        except Exception:
            pass

    # --------------------------------------------------------
    # Label inspection
    # --------------------------------------------------------

    for label_path in labels:

        try:
            lines = label_path.read_text(
                encoding="utf-8"
            ).splitlines()
        except Exception as e:
            invalid_annotations.append(
                (str(label_path), f"Read error: {e}")
            )
            continue

        for line_number, line in enumerate(lines, start=1):

            line = line.strip()

            if not line:
                continue

            values = line.split()

            try:
                class_id = int(values[0])
            except Exception:
                invalid_annotations.append(
                    (str(label_path), line_number, "Invalid class ID")
                )
                continue

            # Class validation
            if class_id not in CLASS_NAMES:
                invalid_annotations.append(
                    (
                        str(label_path),
                        line_number,
                        f"Unknown class ID: {class_id}"
                    )
                )
            else:
                class_counter[class_id] += 1
                split_class_counter[split][class_id] += 1

            # -----------------------------------------------
            # Determine annotation type
            # -----------------------------------------------

            coordinate_values = values[1:]

            # Standard YOLO bounding box
            if len(values) == 5:

                x, y, w, h = map(float, coordinate_values)

                coords = [x, y, w, h]

                if any(v < 0 or v > 1 for v in coords):
                    out_of_range.append(
                        (
                            str(label_path),
                            line_number,
                            "Bounding-box coordinate outside 0-1"
                        )
                    )

            # YOLO segmentation polygon
            elif len(values) >= 7 and len(coordinate_values) % 2 == 0:

                coords = list(map(float, coordinate_values))

                if any(v < 0 or v > 1 for v in coords):
                    out_of_range.append(
                        (
                            str(label_path),
                            line_number,
                            "Segmentation coordinate outside 0-1"
                        )
                    )

                points = len(coordinate_values) // 2
                polygon_point_counter[points] += 1

            else:

                invalid_annotations.append(
                    (
                        str(label_path),
                        line_number,
                        f"Unexpected number of values: {len(values)}"
                    )
                )

            total_annotations += 1


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

duplicate_groups = {
    h: paths
    for h, paths in duplicate_hashes.items()
    if len(paths) > 1
}

# ============================================================
# REPORT
# ============================================================

print("\n")
print("=" * 70)
print("FINAL DATASET AUDIT REPORT")
print("=" * 70)

print("\nDATASET TOTALS")
print("-" * 70)

print(f"Total images       : {total_images}")
print(f"Total label files  : {total_labels}")
print(f"Total annotations  : {total_annotations}")

print("\nIMAGE / LABEL MATCHING")
print("-" * 70)

print(f"Missing labels     : {len(missing_labels)}")
print(f"Missing images     : {len(missing_images)}")

print("\nIMAGE QUALITY")
print("-" * 70)

print(f"Corrupt images     : {len(corrupt_images)}")

print("\nANNOTATION QUALITY")
print("-" * 70)

print(f"Invalid annotations: {len(invalid_annotations)}")
print(f"Out-of-range coords: {len(out_of_range)}")

print("\nCLASS DISTRIBUTION")
print("-" * 70)

for class_id in sorted(CLASS_NAMES):

    name = CLASS_NAMES[class_id]

    print(
        f"{class_id}: {name:<25} "
        f"{class_counter[class_id]:>6} objects"
    )

print("\nCLASS DISTRIBUTION BY SPLIT")
print("-" * 70)

for split in SPLITS:

    print(f"\n{split.upper()}")

    for class_id in sorted(CLASS_NAMES):

        print(
            f"  {class_id}: "
            f"{CLASS_NAMES[class_id]:<25} "
            f"{split_class_counter[split][class_id]:>6}"
        )

print("\nIMAGE DIMENSIONS")
print("-" * 70)

for (width, height), count in dimension_counter.most_common(20):

    print(
        f"{width} x {height}: {count} images"
    )

print("\nPOLYGON POINT DISTRIBUTION")
print("-" * 70)

if polygon_point_counter:

    for points, count in sorted(
        polygon_point_counter.items()
    ):
        print(
            f"{points} points: {count} annotations"
        )

else:

    print("No segmentation polygons detected.")

print("\nDUPLICATE IMAGE GROUPS")
print("-" * 70)

print(
    f"Duplicate groups: {len(duplicate_groups)}"
)

if duplicate_groups:

    for i, paths in enumerate(
        duplicate_groups.values(),
        start=1
    ):

        print(f"\nDuplicate group {i}:")

        for path in paths:
            print(f"  {path}")

print("\n")
print("=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)