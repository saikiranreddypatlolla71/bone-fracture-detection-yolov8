from pathlib import Path
from collections import Counter


DATASET = Path("dataset_detection")

CLASS_NAMES = [
    "elbow positive",
    "fingers positive",
    "forearm fracture",
    "humerus fracture",
    "humerus",
    "shoulder fracture",
    "wrist positive",
]


def analyze_split(split):

    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    images = list(image_dir.glob("*"))
    labels = list(label_dir.glob("*.txt"))

    class_counts = Counter()
    images_with_objects = 0
    background_images = 0
    total_objects = 0

    for image_path in images:

        label_path = label_dir / (image_path.stem + ".txt")

        if not label_path.exists():
            continue

        lines = [
            line.strip()
            for line in label_path.read_text().splitlines()
            if line.strip()
        ]

        if len(lines) == 0:
            background_images += 1
            continue

        images_with_objects += 1

        for line in lines:

            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            class_counts[class_id] += 1
            total_objects += 1

    return {
        "images": len(images),
        "labels": len(labels),
        "images_with_objects": images_with_objects,
        "background_images": background_images,
        "objects": total_objects,
        "class_counts": class_counts,
    }


print("=" * 80)
print("BONE FRACTURE DETECTION - DATASET DISTRIBUTION ANALYSIS")
print("=" * 80)

all_counts = Counter()

for split in ["train", "valid", "test"]:

    result = analyze_split(split)

    print()
    print("-" * 80)
    print(f"{split.upper()}")
    print("-" * 80)

    print(f"Images              : {result['images']}")
    print(f"Label files         : {result['labels']}")
    print(f"Images with objects : {result['images_with_objects']}")
    print(f"Background images   : {result['background_images']}")
    print(f"Objects             : {result['objects']}")

    print()
    print("Class distribution:")

    for class_id, class_name in enumerate(CLASS_NAMES):

        count = result["class_counts"][class_id]
        all_counts[class_id] += count

        print(f"{class_id}: {class_name:<22} {count}")

print()
print("=" * 80)
print("TOTAL CLASS DISTRIBUTION")
print("=" * 80)

for class_id, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_id}: {class_name:<22} "
        f"{all_counts[class_id]}"
    )

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)