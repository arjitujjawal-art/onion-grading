"""Train YOLOv8 segmentation model on the onion grading dataset.
Optimized for CPU training: uses small model (yolov8s-seg), reduced epochs,
and aggressive caching.
"""
from __future__ import annotations
import sys
sys.path.insert(0, ".")

from ultralytics import YOLO
from pathlib import Path

def main():
    data_yaml = "data/real/onion.yaml"
    model_dir = Path("ml-backend/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Use yolov8s-seg.pt as base — small enough for CPU, good accuracy
    model = YOLO("yolov8s-seg.pt")

    model.train(
        data=data_yaml,
        epochs=50,
        batch=8,           # small batch for CPU
        imgsz=512,
        name="onion_seg",
        workers=0,         # no multiprocessing on Windows
        cache=True,        # cache images for faster CPU training
        device="cpu",
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        mask_ratio=4,      # segmentation mask downsampling
        max_det=20,        # max onions per image
        project=str(model_dir),
        exist_ok=True,
        verbose=True,
        augment=True,      # use built-in augmentations (mosaic, hsv, flip, scale)
        # Augment severity for small dataset generalization
        degrees=0.0,
        translate=0.1,
        scale=0.2,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.0,
    )

    # Export to ONNX
    best = model_dir / "runs" / "onion_seg" / "weights" / "best.pt"
    if best.exists():
        model.export(format="onnx", weights=str(best))
        print(f"\nModel exported to ONNX from {best}")

if __name__ == "__main__":
    main()
