from ultralytics import YOLO
import time
import statistics


def main():
    model_path = r"runs\detect\runs\experiments\yolov8n_768epoch50\weights\best.pt"
    image_dir = r"dataset_detection\test\images"

    model = YOLO(model_path)

    print("=" * 70)
    print("YOLOv8n 768x768 INFERENCE BENCHMARK")
    print("=" * 70)

    print("Warming up GPU...")

    # GPU warm-up
    for _ in range(10):
        model.predict(
            source=image_dir,
            imgsz=768,
            batch=1,
            device=0,
            workers=0,
            conf=0.20,
            verbose=False,
            max_det=100
        )

    print("Warm-up complete.")
    print()

    # Benchmark
    times = []

    print("Running benchmark...")
    
    for _ in range(3):
        start = time.perf_counter()

        model.predict(
            source=image_dir,
            imgsz=768,
            batch=1,
            device=0,
            workers=0,
            conf=0.20,
            verbose=False,
            max_det=100
        )

        end = time.perf_counter()

        elapsed = end - start
        times.append(elapsed)

        print(f"Run time: {elapsed:.3f} seconds")

    average_time = statistics.mean(times)

    # Number of test images
    import os

    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    )

    image_count = sum(
        1
        for f in os.listdir(image_dir)
        if f.lower().endswith(image_extensions)
    )

    seconds_per_image = average_time / image_count
    fps = 1 / seconds_per_image
    ms_per_image = seconds_per_image * 1000

    print()
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    print(f"Test images       : {image_count}")
    print(f"Average run time  : {average_time:.3f} seconds")
    print(f"Latency/image     : {ms_per_image:.2f} ms")
    print(f"Images/second     : {fps:.2f} FPS")

    print("=" * 70)


if __name__ == "__main__":
    main()