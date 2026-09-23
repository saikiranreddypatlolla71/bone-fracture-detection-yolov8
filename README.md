# Bone Fracture Detection Using YOLO

## Computer Vision Project — Object Detection on X-Ray Images

An end-to-end computer vision project for detecting annotated bone/fracture-related regions in X-ray images using the YOLO object-detection framework.

> **Important:** This is an academic computer-vision project. The model evaluates agreement between predictions and dataset annotations and must not be interpreted as a clinically validated diagnostic system.

---

## 1. Project Overview

Bone fracture detection from X-ray images is an important computer-vision problem involving object detection and localization.

This project develops and evaluates a YOLO-based object-detection pipeline for identifying annotated bone/fracture-related regions in X-ray images.

The project covers the complete workflow:

- Dataset inspection
- Dataset quality validation
- Class-distribution analysis
- Source-level leakage investigation
- Source-grouped dataset splitting
- Segmentation-to-detection annotation conversion
- YOLO dataset preparation
- Baseline model training
- Image-resolution experiments
- Model-size comparison
- X-ray-specific augmentation experiment
- Independent test evaluation
- Confidence-threshold analysis
- Quantitative error analysis
- Qualitative error visualization
- GPU inference benchmarking
- Project documentation

---

# 2. Project Objectives

The main objectives of this project are:

1. Prepare and validate an X-ray object-detection dataset.
2. Investigate possible source-level data leakage.
3. Reduce source-level leakage during dataset splitting.
4. Convert segmentation polygon annotations into detection bounding boxes.
5. Train YOLO-based object-detection models.
6. Compare different image resolutions and model sizes.
7. Evaluate models using Precision, Recall, mAP@50, and mAP@50–95.
8. Perform confidence-threshold analysis.
9. Perform quantitative and qualitative error analysis.
10. Measure inference performance on GPU hardware.
11. Document limitations and possible future improvements.

---

# 3. Dataset

The project uses the provided `BoneFractureYolo8` dataset.

The original dataset contained:

- **4,148 images**
- **7 classes**
- YOLO-format segmentation polygon annotations

## Classes

| Class ID | Class Name |
|---:|---|
| 0 | elbow positive |
| 1 | fingers positive |
| 2 | forearm fracture |
| 3 | humerus fracture |
| 4 | humerus |
| 5 | shoulder fracture |
| 6 | wrist positive |

---

# 4. Dataset Preparation

The original dataset was preserved without modification.

A separate detection dataset was created at:

```text
dataset_detection/
````

The prepared dataset has the following structure:

```text
dataset_detection/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

## Prepared Dataset Split

| Split      |    Images | Label Files |
| ---------- | --------: | ----------: |
| Train      |     3,345 |       3,345 |
| Validation |       415 |         415 |
| Test       |       388 |         388 |
| **Total**  | **4,148** |   **4,148** |

The prepared detection dataset contains:

**2,388 annotated objects.**

---

# 5. Source-Level Leakage Investigation

The original dataset was investigated for possible cross-split source overlap.

An exact image-hash comparison did not identify exact duplicate image files across the original train, validation, and test partitions.

However, filename/source analysis identified repeated apparent source identifiers across the original partitions.

Because patient-level identifiers were not available, true patient-level independence could not be directly verified.

To reduce potential source-level contamination, images sharing the same identifiable source identifier were grouped before dataset splitting.

A fixed random seed of:

```text
42
```

was used for the source-grouped split.

The resulting dataset was approximately divided into:

```text
80% Training
10% Validation
10% Testing
```

> **Note:** Source-level grouping reduces identifiable source overlap but does not prove patient-level independence.

---

# 6. Annotation Format

The original annotations were segmentation polygons rather than standard YOLO detection labels.

The original annotation format contained:

```text
class_id x1 y1 x2 y2 ...
```

where the coordinate pairs represented polygon points.

For the object-detection experiments, each polygon was converted into an enclosing rectangular bounding box.

The resulting YOLO detection format is:

```text
class_id x_center y_center width height
```

with normalized coordinates.

## Important Annotation Limitation

The polygon-to-bounding-box conversion removes detailed segmentation boundary information.

Therefore, the trained models perform **object detection using rectangular bounding boxes**, rather than segmentation.

---

# 7. Dataset Validation

The prepared dataset was validated for:

* Image/label pairing
* Label-file validity
* Correct number of annotation values
* Valid class IDs
* Normalized coordinates
* Positive bounding-box dimensions
* Bounding boxes within image boundaries

Final validation result:

```text
Images       : 4148
Label files  : 4148
Objects      : 2388
Errors       : 0

ALL CHECKS PASSED
```

---

# 8. Dataset Class Distribution

The complete prepared dataset contained 2,388 annotated objects.

| Class             |   Objects |
| ----------------- | --------: |
| elbow positive    |       385 |
| fingers positive  |       606 |
| forearm fracture  |       373 |
| humerus fracture  |         3 |
| humerus           |       362 |
| shoulder fracture |       397 |
| wrist positive    |       262 |
| **Total**         | **2,388** |

## Rare Class Limitation

The `humerus fracture` class contains only:

```text
3 annotated objects
```

in the complete dataset.

After source-level grouping and splitting, all three objects were assigned to the training set.

Therefore:

```text
Validation instances = 0
Test instances       = 0
```

Independent validation/test performance for this class cannot be calculated.

This class limitation must be considered when interpreting the overall results.

---

# 9. Background Images

A substantial proportion of images contained no annotated objects.

| Split      | Background Images | Total Images |
| ---------- | ----------------: | -----------: |
| Train      |             1,682 |        3,345 |
| Validation |               216 |          415 |
| Test       |               190 |          388 |

Approximate background-image proportions were:

* Training: **50.3%**
* Validation: **52.0%**
* Test: **49.0%**

---

# 10. Software Environment

The project was developed using:

```text
Python 3.11.9
PyTorch 2.14.0+cu130
Ultralytics 8.4.159
OpenCV 5.0.0
NumPy 2.4.6
Pandas 3.0.6
Matplotlib 3.11.2
Pillow 12.3.0
scikit-learn 1.9.1
PyYAML 6.0.3
```

## Hardware

```text
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
VRAM: 6 GB
```

PyTorch successfully detected and used the NVIDIA GPU.

---

# 11. Model Training Experiments

Several YOLO experiments were completed.

| Experiment                  | Model   | Image Size | Maximum Epochs | Batch | Val Precision | Val Recall | Val mAP@50 | Val mAP@50–95 |
| --------------------------- | ------- | ---------: | -------------: | ----: | ------------: | ---------: | ---------: | ------------: |
| Baseline                    | YOLOv8n |        640 |             10 |     8 |         0.104 |      0.183 |      0.139 |        0.0564 |
| Extended Training           | YOLOv8n |        640 |             50 |     8 |         0.189 |      0.302 |      0.209 |        0.0922 |
| Higher Resolution           | YOLOv8n |        768 |             50 |     4 |     **0.261** |  **0.280** |  **0.245** |     **0.106** |
| X-ray-Specific Augmentation | YOLOv8n |        768 |             50 |     4 |         0.211 |      0.177 |      0.138 |        0.0507 |
| Larger Model                | YOLOv8s |        768 |             50 |     4 |        0.2212 |     0.2558 |     0.2188 |        0.0903 |

### Training notes

* The YOLOv8n 640×640 extended experiment had a maximum of 50 epochs but **early-stopped after 37 epochs**.
* The YOLOv8n 768×768 standard experiment completed 50 epochs.
* The YOLOv8n 768×768 X-ray-specific augmentation experiment completed 50 epochs.
* The YOLOv8s 768×768 experiment completed 50 epochs.
* Windows multiprocessing was configured with `workers=0` for reliable execution.

---

# 12. X-Ray-Specific Augmentation Experiment

A dedicated augmentation experiment was performed at 768×768 using YOLOv8n.

The experiment used conservative transformations intended to avoid unrealistic geometric changes to X-ray anatomy.

The experiment achieved:

```text
Validation Precision : 0.211
Validation Recall    : 0.177
Validation mAP@50    : 0.138
Validation mAP@50-95 : 0.0507
```

These validation results were lower than the standard YOLOv8n 768×768 experiment.

The augmentation experiment was therefore retained as an experimental result rather than used as the selected model.

---

# 13. Selected Model

The selected model was:

```text
YOLOv8n
Input resolution: 768 × 768
Batch size: 4
Maximum epochs: 50
Random seed: 42
```

Model weights:

```text
runs\detect\runs\experiments\yolov8n_768epoch50\weights\best.pt
```

The model was selected based on the completed validation experiments.

## Validation Performance

| Metric    |    Result |
| --------- | --------: |
| Precision |     0.261 |
| Recall    |     0.280 |
| mAP@50    | **0.245** |
| mAP@50–95 | **0.106** |

---

# 14. Independent Test Evaluation

The selected YOLOv8n 768×768 model was evaluated on the held-out test set.

Test set:

```text
Images      : 388
Instances   : 224
```

## Standard Test Evaluation

| Metric    |     Result |
| --------- | ---------: |
| Precision |  **0.226** |
| Recall    |  **0.320** |
| mAP@50    |  **0.205** |
| mAP@50–95 | **0.0834** |

These are the **primary independent test results** for the selected model.

---

# 15. Confidence-Threshold Analysis

A confidence-threshold sweep was performed on the validation set.

The evaluated thresholds were:

```text
0.10
0.15
0.20
0.25
0.30
0.35
0.40
0.45
0.50
0.55
0.60
```

The highest validation F1-score occurred at:

```text
Confidence threshold = 0.20
```

## Validation Operating Point at Confidence 0.20

| Metric    |     Result |
| --------- | ---------: |
| Precision |     0.3104 |
| Recall    |     0.2333 |
| F1-score  | **0.2664** |
| mAP@50    |     0.1977 |
| mAP@50–95 |     0.0881 |

The threshold was selected using the validation set and then locked before evaluating the held-out test set at this operating point.

---

# 16. Test Evaluation at Locked Confidence Threshold

Using:

```text
Confidence threshold = 0.20
```

the held-out test results were:

| Metric    | Result |
| --------- | -----: |
| Precision |  0.246 |
| Recall    |  0.240 |
| mAP@50    |  0.152 |
| mAP@50–95 | 0.0616 |

The threshold-selected evaluation is reported separately from the standard test evaluation.

The confidence threshold did not improve the held-out test mAP values compared with the standard evaluation.

---

# 17. Quantitative Error Analysis

A custom test-set error analysis was performed using:

```text
Confidence threshold = 0.20
IoU matching threshold = 0.50
```

## Overall Results

| Measure              | Count |
| -------------------- | ----: |
| Test images          |   388 |
| Ground-truth objects |   224 |
| Predictions          |   191 |
| True positives       |    51 |
| False positives      |   140 |
| False negatives      |   173 |

Custom error-analysis metrics:

```text
Precision = 0.2670
Recall    = 0.2277
F1-score  = 0.2458
```

## Per-Class Error Analysis

| Class             | TP | FP | FN | Precision | Recall |
| ----------------- | -: | -: | -: | --------: | -----: |
| elbow positive    |  1 | 16 | 37 |    0.0588 | 0.0263 |
| fingers positive  | 10 | 32 | 46 |    0.2381 | 0.1786 |
| forearm fracture  | 13 | 15 | 13 |    0.4643 | 0.5000 |
| humerus fracture  |  0 |  0 |  0 |       N/A |    N/A |
| humerus           | 13 | 34 | 17 |    0.2766 | 0.4333 |
| shoulder fracture | 13 | 35 | 31 |    0.2708 | 0.2955 |
| wrist positive    |  1 |  8 | 29 |    0.1111 | 0.0333 |

The `forearm fracture` class showed the strongest performance in this fixed-threshold error analysis.

The `elbow positive` and `wrist positive` classes showed particularly high false-negative rates.

> **Metric note:** The custom TP/FP/FN analysis uses explicit one-to-one matching at IoU 0.50 and should be treated as a detailed error-analysis tool. The primary model-performance metrics remain the official Ultralytics evaluation metrics reported above.

---

# 18. Qualitative Error Analysis

Prediction and ground-truth comparison images were generated from representative test images.

The visual analysis demonstrated:

* Successful detections
* Missed detections
* Moderate-confidence predictions
* Imperfect localization
* Background/negative examples

A representative successful example showed a `fingers positive` prediction with confidence approximately 0.35.

A representative missed-detection example involved an `elbow positive` ground-truth annotation with no corresponding prediction.

The qualitative findings were consistent with the quantitative error analysis.

The visualizations compare model predictions with dataset annotations and do not constitute clinical interpretation of the X-ray images.

---

# 19. Inference Performance

Inference performance was benchmarked using the selected YOLOv8n model on the NVIDIA RTX 3050 6 GB GPU.

Configuration:

```text
Model: YOLOv8n
Input resolution: 768 × 768
Batch size: 1
Test images: 388
```

## Benchmark Results

| Metric            |               Result |
| ----------------- | -------------------: |
| Average run time  |        3.820 seconds |
| Latency per image |              9.85 ms |
| Throughput        | 101.57 images/second |

These values represent the measured performance under the specific hardware, software, image-size, and benchmarking conditions used in this project.

---

# 20. Project Structure

The main project structure is:

```text
bone-fracture-detection/
│
├── .venv/
│
├── dataset_detection/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
│
├── annotation_preview/
│
├── runs/
│   └── detect/
│       └── runs/
│           ├── baseline/
│           ├── experiments/
│           └── evaluation/
│
├── prepare_detection_dataset.py
├── validate_detection_dataset.py
├── analyze_detection_dataset.py
├── train_baseline.py
├── train_yolov8n_50.py
├── train_yolov8n_768.py
├── train_yolov8n_768_xray_aug.py
├── train_yolov8s_768.py
├── confidence_threshold_analysis.py
├── benchmark_inference.py
├── explainability_visualization.py
├── test_error_analysis.py
├── final_experiment_summary.py
│
├── final_experiment_summary.csv
└── README.md
```

---

# 21. Important Output Files

## Selected Model

```text
runs\detect\runs\experiments\yolov8n_768epoch50\weights\best.pt
```

## Standard Test Evaluation

```text
runs\detect\runs\evaluation\yolov8n_768_test_final
```

## Confidence Threshold Analysis

```text
runs\evaluation\confidence_threshold\
```

## Locked-Threshold Test Evaluation

```text
runs\detect\runs\evaluation\yolov8n_768_test_conf020-2
```

## Explainability Visualizations

```text
runs\evaluation\explainability\
```

## Quantitative Error Analysis

```text
runs\evaluation\error_analysis\
```

## Final Experiment Summary

```text
runs\evaluation\final_experiment_summary.csv
runs\evaluation\FINAL_EXPERIMENT_SUMMARY.md
```

---

# 22. Reproducibility

The project uses fixed random seeds where applicable.

Primary experimental seed:

```text
42
```

For the Windows environment, YOLO training and evaluation were configured with:

```text
workers = 0
```

to avoid multiprocessing-related execution issues.

The original dataset was preserved separately from the prepared detection dataset.

The prepared dataset is therefore reproducible without modifying the original source data.

---

# 23. Limitations

## 23.1 Class imbalance

The `humerus fracture` class contains only three annotated objects in the complete dataset.

All three were assigned to the training partition after source-level grouping.

Therefore, there are no validation or test examples for independent evaluation of this class.

## 23.2 Patient-level information

The dataset did not provide sufficient patient-level identifiers to directly verify patient-level independence.

Source-level grouping based on identifiable filename/source information was therefore used as a leakage-reduction strategy.

## 23.3 Annotation conversion

The original dataset contained segmentation polygons.

These were converted into enclosing rectangular bounding boxes.

This conversion loses detailed segmentation information.

## 23.4 Class imbalance

The number of examples differs substantially between classes.

This can affect model learning and class-specific performance.

## 23.5 Background images

Approximately half of the images in each split contain no annotated objects.

This influences the detection task and error profile.

## 23.6 Model performance

The independent test results show substantial false negatives and false-positive predictions.

The model therefore requires further development before any practical application could be considered.

## 23.7 Clinical validation

This project does not establish clinical diagnostic accuracy, clinical sensitivity, clinical specificity, or suitability for medical decision-making.

---

# 24. Future Improvements

Potential future work includes:

* Increasing the number of examples for rare classes.
* Obtaining additional diverse X-ray datasets.
* Obtaining reliable patient-level identifiers for stronger leakage control.
* Performing patient-level dataset splitting.
* Improving annotation quality.
* Comparing detection with segmentation models.
* Testing additional YOLO architectures.
* Hyperparameter optimization.
* Investigating class-imbalance strategies.
* Testing additional augmentation strategies.
* Performing cross-validation where appropriate.
* Performing external-dataset validation.
* Improving confidence calibration.
* Expanding error-mining analysis.
* Investigating more rigorous model-explainability methods.
* Developing a research/demo interface using Streamlit or Gradio.

---

# 25. Conclusion

This project developed an end-to-end YOLO-based computer-vision pipeline for detecting annotated bone/fracture-related regions in X-ray images.

The selected model was:

```text
YOLOv8n
768 × 768 input resolution
```

The primary independent test-set performance was:

```text
Precision : 0.226
Recall    : 0.320
mAP@50    : 0.205
mAP@50-95 : 0.0834
```

The model showed stronger performance for some classes, particularly `forearm fracture`, while several other classes showed substantial missed detections.

The project also demonstrated the importance of:

* Dataset inspection
* Source-level leakage analysis
* Class-distribution analysis
* Annotation validation
* Independent testing
* Confidence-threshold analysis
* Quantitative error analysis
* Qualitative visualization
* Inference benchmarking

The results should be interpreted in the context of the dataset limitations, particularly severe class imbalance and the absence of validation/test examples for the `humerus fracture` class.

---

# 26. Disclaimer

This project is intended for academic and educational computer-vision research.

The model predictions represent computer-vision outputs evaluated against dataset annotations.

They must not be interpreted as medical diagnoses.

The system has not been clinically validated and must not be used to make clinical decisions.

---

# Author

**Patlolla Saikiran Reddy**

PharmD Student
MNR College of Pharmacy
Telangana, India

---

# License

The project should comply with the license and attribution requirements of the dataset and any third-party software or pretrained models used.

````