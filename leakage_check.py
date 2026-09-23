from pathlib import Path
from PIL import Image
import hashlib
from collections import defaultdict

DATASET = Path(r"C:\Users\saiki\Downloads\archive (2)\BoneFractureYolo8")

SPLITS = ["train", "valid", "test"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def md5_hash(path):
    h = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


# ------------------------------------------------------------
# Collect images
# ------------------------------------------------------------

all_images = []

for split in SPLITS:

    folder = DATASET / split / "images"

    for path in folder.rglob("*"):

        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            all_images.append((split, path))


print("=" * 70)
print("CROSS-SPLIT DUPLICATE / LEAKAGE CHECK")
print("=" * 70)

print(f"\nTotal images checked: {len(all_images)}")


# ------------------------------------------------------------
# Hash images
# ------------------------------------------------------------

hash_groups = defaultdict(list)

for split, path in all_images:

    try:
        h = md5_hash(path)
        hash_groups[h].append((split, path))
    except Exception as e:
        print(f"Could not hash: {path}")
        print(e)


# ------------------------------------------------------------
# Find exact duplicate groups
# ------------------------------------------------------------

duplicate_groups = []

for h, items in hash_groups.items():

    if len(items) > 1:

        splits = {x[0] for x in items}

        if len(splits) > 1:
            duplicate_groups.append((h, items))


# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(
    f"\nCross-split exact duplicate groups: "
    f"{len(duplicate_groups)}"
)

if duplicate_groups:

    print("\nWARNING: Exact duplicate images exist across splits.")

    for i, (h, items) in enumerate(
        duplicate_groups,
        start=1
    ):

        print(f"\nDuplicate group {i}:")

        for split, path in items:

            print(f"  [{split}] {path}")

else:

    print("\nNo exact duplicate images were found across splits.")


# ------------------------------------------------------------
# Filename similarity check
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FILENAME / SOURCE PATTERN CHECK")
print("=" * 70)

for split, path in all_images[:10]:

    print(f"[{split}] {path.name}")

print("\nCheck complete.")