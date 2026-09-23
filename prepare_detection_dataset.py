from pathlib import Path
from collections import defaultdict, Counter
import shutil
import random
import re

# ============================================================
# CONFIGURATION
# ============================================================

SOURCE = Path(
    r"C:\Users\saiki\Downloads\archive (2)\BoneFractureYolo8"
)

OUTPUT = Path(
    r"C:\Users\saiki\bone-fracture-detection\dataset_detection"
)

SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

CLASS_NAMES = [
    "elbow positive",
    "fingers positive",
    "forearm fracture",
    "humerus fracture",
    "humerus",
    "shoulder fracture",
    "wrist positive",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

random.seed(SEED)


# ============================================================
# HELPERS
# ============================================================

def source_id(filename):
    """
    Remove Roboflow's .rf.<hash> suffix.

    Example:
    image1_125_png.rf.abcdef123.jpg
    ->
    image1_125_png
    """

    stem = Path(filename).stem

    return re.sub(
        r"\.rf\.[a-fA-F0-9]+$",
        "",
        stem
    )


def polygon_to_bbox(coords):
    """
    Convert normalized polygon coordinates:

    x1 y1 x2 y2 x3 y3 ...

    into:

    x_center y_center width height
    """

    xs = coords[0::2]
    ys = coords[1::2]

    xmin = min(xs)
    xmax = max(xs)

    ymin = min(ys)
    ymax = max(ys)

    width = xmax - xmin
    height = ymax - ymin

    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2

    return (
        x_center,
        y_center,
        width,
        height,
    )


def find_image(split, stem):

    image_dir = SOURCE / split / "images"

    for ext in IMAGE_EXTENSIONS:

        candidate = image_dir / (stem + ext)

        if candidate.exists():
            return candidate

    return None


# ============================================================
# COLLECT ALL DATA
# ============================================================

records = []

for original_split in ["train", "valid", "test"]:

    image_dir = SOURCE / original_split / "images"
    label_dir = SOURCE / original_split / "labels"

    for image_path in image_dir.iterdir():

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = label_dir / (image_path.stem + ".txt")

        if not label_path.exists():
            print(
                "WARNING: missing label:",
                image_path.name
            )
            continue

        records.append({
            "image": image_path,
            "label": label_path,
            "original_split": original_split,
            "source": source_id(image_path.name),
        })


print("=" * 70)
print("SOURCE DATA")
print("=" * 70)

print("Images discovered:", len(records))


# ============================================================
# GROUP BY SOURCE
# ============================================================

groups = defaultdict(list)

for record in records:

    groups[record["source"]].append(record)


print("Unique source groups:", len(groups))


# ============================================================
# CHECK CLASS CONTENT OF EACH GROUP
# ============================================================

for source, items in groups.items():

    group_classes = set()

    for item in items:

        for line in item["label"].read_text(
            encoding="utf-8"
        ).splitlines():

            values = line.split()

            if values:
                group_classes.add(
                    int(values[0])
                )

    for item in items:
        item["classes"] = group_classes


# ============================================================
# SPLIT SOURCE GROUPS
# ============================================================

source_list = list(groups.keys())

random.shuffle(source_list)

n = len(source_list)

train_end = int(n * TRAIN_RATIO)

val_end = train_end + int(n * VAL_RATIO)

train_sources = set(
    source_list[:train_end]
)

val_sources = set(
    source_list[train_end:val_end]
)

test_sources = set(
    source_list[val_end:]
)


print("\n" + "=" * 70)
print("NEW SOURCE-LEVEL SPLIT")
print("=" * 70)

print("Train source groups:", len(train_sources))
print("Validation source groups:", len(val_sources))
print("Test source groups:", len(test_sources))


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

if OUTPUT.exists():

    print("\nRemoving previous output dataset...")

    shutil.rmtree(OUTPUT)


for split in ["train", "valid", "test"]:

    (OUTPUT / split / "images").mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT / split / "labels").mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# DETERMINE DESTINATION SPLIT
# ============================================================

def destination_split(source):

    if source in train_sources:
        return "train"

    if source in val_sources:
        return "valid"

    return "test"


# ============================================================
# CONVERT POLYGONS TO BOUNDING BOXES
# ============================================================

stats = Counter()

annotation_count = Counter()

invalid_count = 0


for record in records:

    source = record["source"]

    split = destination_split(source)

    image_src = record["image"]
    label_src = record["label"]

    image_dst = (
        OUTPUT /
        split /
        "images" /
        image_src.name
    )

    label_dst = (
        OUTPUT /
        split /
        "labels" /
        (image_src.stem + ".txt")
    )

    # --------------------------------------------------------
    # Copy image
    # --------------------------------------------------------

    shutil.copy2(
        image_src,
        image_dst
    )

    # --------------------------------------------------------
    # Convert annotation
    # --------------------------------------------------------

    output_lines = []

    for line in label_src.read_text(
        encoding="utf-8"
    ).splitlines():

        line = line.strip()

        if not line:
            continue

        values = line.split()

        try:

            class_id = int(values[0])

            coords = [
                float(x)
                for x in values[1:]
            ]

        except Exception:

            invalid_count += 1
            continue

        # ----------------------------------------------------
        # Detection bbox already
        # ----------------------------------------------------

        if len(values) == 5:

            x_center = coords[0]
            y_center = coords[1]
            width = coords[2]
            height = coords[3]

        # ----------------------------------------------------
        # Segmentation polygon
        # ----------------------------------------------------

        elif (
            len(values) >= 7
            and len(coords) % 2 == 0
        ):

            (
                x_center,
                y_center,
                width,
                height,
            ) = polygon_to_bbox(coords)

        else:

            invalid_count += 1
            continue

        # ----------------------------------------------------
        # Validate bbox
        # ----------------------------------------------------

        bbox = [
            x_center,
            y_center,
            width,
            height,
        ]

        if any(
            x < 0 or x > 1
            for x in bbox
        ):

            invalid_count += 1
            continue

        if width <= 0 or height <= 0:

            invalid_count += 1
            continue

        output_lines.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

        annotation_count[split] += 1

        stats[
            (split, class_id)
        ] += 1

    label_dst.write_text(
        "\n".join(output_lines),
        encoding="utf-8"
    )


# ============================================================
# WRITE DATA.YAML
# ============================================================

yaml_text = f"""path: {OUTPUT.as_posix()}

train: train/images
val: valid/images
test: test/images

nc: {len(CLASS_NAMES)}

names:
"""

for i, name in enumerate(CLASS_NAMES):

    yaml_text += f"  {i}: {name}\n"


(OUTPUT / "data.yaml").write_text(
    yaml_text,
    encoding="utf-8"
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("CONVERSION COMPLETE")
print("=" * 70)

for split in ["train", "valid", "test"]:

    images = list(
        (OUTPUT / split / "images").iterdir()
    )

    labels = list(
        (OUTPUT / split / "labels").glob("*.txt")
    )

    print(
        f"{split:>6}: "
        f"{len(images)} images | "
        f"{len(labels)} labels"
    )


print("\nAnnotations by split:")

for split in ["train", "valid", "test"]:

    print(
        f"{split:>6}: "
        f"{annotation_count[split]}"
    )


print("\nClass distribution:")

for class_id, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_id}: "
        f"{class_name:<25} "
        f"train={stats[('train', class_id)]} "
        f"val={stats[('valid', class_id)]} "
        f"test={stats[('test', class_id)]}"
    )


print("\nInvalid annotations:", invalid_count)

print("\nDataset created at:")

print(OUTPUT)

print("\nDATASET PREPARATION FINISHED")