import sys
import platform

print("=" * 70)
print("BONE FRACTURE DETECTION - ENVIRONMENT CHECK")
print("=" * 70)

print("\nSYSTEM")
print("-" * 70)
print("Operating System :", platform.platform())
print("Python           :", sys.version)

print("\nPYTORCH")
print("-" * 70)

import torch

print("PyTorch          :", torch.__version__)
print("CUDA available   :", torch.cuda.is_available())
print("PyTorch CUDA     :", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU count        :", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        gpu = torch.cuda.get_device_properties(i)

        print(f"\nGPU {i}")
        print("Name             :", torch.cuda.get_device_name(i))
        print(f"VRAM             : {gpu.total_memory / (1024**3):.2f} GB")

print("\nULTRALYTICS")
print("-" * 70)

import ultralytics

print("Ultralytics      :", ultralytics.__version__)

print("\nCOMPUTER VISION")
print("-" * 70)

import cv2
import numpy
import pandas
import matplotlib
import PIL
import sklearn
import tqdm
import yaml

print("OpenCV           :", cv2.__version__)
print("NumPy            :", numpy.__version__)
print("Pandas           :", pandas.__version__)
print("Matplotlib       :", matplotlib.__version__)
print("Pillow           :", PIL.__version__)
print("Scikit-learn     :", sklearn.__version__)
print("PyYAML           :", yaml.__version__)

print("\n" + "=" * 70)
print("ENVIRONMENT CHECK COMPLETE")
print("=" * 70)