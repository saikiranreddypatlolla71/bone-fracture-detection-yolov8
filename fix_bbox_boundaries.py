from pathlib import Path

DATASET = Path("dataset_detection")

fixed = 0

for split in ["train", "valid", "test"]:
    label_dir = DATASET / split / "labels"

    for label_file in label_dir.glob("*.txt"):
        lines = label_file.read_text().splitlines()
        new_lines = []

        for line in lines:
            if not line.strip():
                continue

            parts = line.split()

            if len(parts) != 5:
                new_lines.append(line)
                continue

            cls, xc, yc, w, h = map(float, parts)

            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2

            original = (x1, y1, x2, y2)

            # Allow only tiny floating-point boundary errors.
            eps = 1e-6

            if x1 < 0 and x1 >= -eps:
                x1 = 0.0

            if y1 < 0 and y1 >= -eps:
                y1 = 0.0

            if x2 > 1 and x2 <= 1 + eps:
                x2 = 1.0

            if y2 > 1 and y2 <= 1 + eps:
                y2 = 1.0

            if original != (x1, y1, x2, y2):
                fixed += 1

            # Recalculate YOLO representation
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            new_lines.append(
                f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
            )

        label_file.write_text("\n".join(new_lines) + "\n")

print("=" * 70)
print("BOUNDARY PRECISION FIX")
print("=" * 70)
print(f"Labels adjusted: {fixed}")
print("Original dataset was NOT modified.")