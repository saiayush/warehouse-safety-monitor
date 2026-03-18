# Warehouse Safety Monitoring using YOLO

## Overview
This project implements a real-time warehouse safety monitoring system using a fine-tuned YOLO model. It detects forklifts, persons, and PPE equipment, and applies rule-based logic to identify unsafe situations such as missing PPE, unsafe proximity to forklifts, and unattended machinery.

The system processes video input, performs multi-class detection, generates danger zones around forklifts, and outputs annotated video along with safety logs.

---

## Features
- Multi-class detection:
  - Forklift, Person, Gloves, Hard Hat, Mask, Safety Boots, Vest
- Dynamic danger zone generation around forklifts
- PPE compliance detection per person
- Safety state classification:
  - CRITICAL_VIOLATION
  - AUTHORIZED_OPERATOR
  - PPE_VIOLATION_SAFE
  - COMPLIANT
  - UNATTENDED_MACHINE
- Annotated video output with:
  - Bounding boxes
  - Alert labels
  - Danger zones
  - HUD (frame stats, FPS)
  - Legend (cumulative counts)
- CSV logging of safety events
- Video-based inference pipeline

---

## Results

### Finetuned Model Performance
mAP@0.5:      0.8269  
mAP@0.5:0.95: 0.5388  
Precision:    0.8279  
Recall:       0.7510  

### Inference Speed
~20 FPS on local webcam/video inference

Note: Detection metrics are evaluated on a validation dataset. FPS depends on hardware.

check out the annotated video at outputs/output_annotated.mp4
---

## Installation

**Clone the repository:** git clone https://github.com/saiayush/warehouse-safety-monitor.git

cd warehouse-safety-monitor

**Install dependencies:** pip install -r requirements.txt

Ensure the path in `inference.py`, 'weights/best.pt' is to the model

**Run Inference:** python inference.py

**Provide video path:** sample_inputs/test_video.mp4

**Outputs:** annotated video at outputs folder and logs

---

**## Future Work**

Improve YOLO performance for small PPE objects

Reduce false positives in cluttered scenes

Add tracking for temporal consistency

Optimize for real-time deployment on edge devices






