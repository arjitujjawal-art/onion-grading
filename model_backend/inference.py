"""
Onion Grading Backend - YOLOv8 Segmentation Inference CLI
Supports both direct model inference and the full metric grading pipeline.

Usage:
    python inference.py --source <image_path>
    python inference.py --source <image_path> --pipeline
    python inference.py --source <folder_path> --save
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO
import cv2
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
CLASS_NAMES = ["grade_a", "grade_urs", "reject"]


def load_model():
    """Load the YOLOv8 segmentation model."""
    return YOLO(MODEL_PATH)


def predict(model, source, conf=0.35, iou=0.6, device=0, save=False):
    """Run inference on an image or directory of images."""
    save_dir = os.path.join(os.path.dirname(MODEL_PATH), "output") if save else None
    results = model(source, conf=conf, iou=iou, device=device, save=save, project=save_dir)
    predictions = []

    for i, result in enumerate(results):
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"

            detection = {
                "image_index": i,
                "class": cls_name,
                "class_id": cls_id,
                "confidence": round(conf_val, 4),
                "bbox": {
                    "x1": float(xyxy[0]),
                    "y1": float(xyxy[1]),
                    "x2": float(xyxy[2]),
                    "y2": float(xyxy[3]),
                },
            }
            predictions.append(detection)

    return results, predictions


def main():
    parser = argparse.ArgumentParser(description="Onion grading inference tool")
    parser.add_argument("--source", type=str, required=True, help="Image or folder path")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (default: 0.35)")
    parser.add_argument("--device", type=str, default="0", help="Device (0 for GPU, cpu for CPU)")
    parser.add_argument("--save", action="store_true", help="Save annotated images to disk")
    parser.add_argument("--pipeline", action="store_true", help="Run full metric grading pipeline (ArUco + Policy)")
    args = parser.parse_args()

    if args.pipeline:
        from ml_backend.pipeline import process_image
        print(f"Running full metric grading pipeline on: {args.source}")
        res = process_image(args.source, conf_threshold=args.conf, generate_artifacts=args.save)
        print("\nPipeline Result:")
        print(f"  Status:       {res['status']}")
        print(f"  Total Onions: {res['summary']['total']}")
        print(f"  Overall:      {res['summary']['overall_status']}")
        for o in res["onions"]:
            print(f"  - {o['onion_id']}: {o['visual_class']} | {o['diameter_mm']} mm | Grade: {o['final_grade']} ({o['confidence']*100:.1f}%) | {o['reason_codes']}")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = load_model()

    results, predictions = predict(model, args.source, conf=args.conf, device=args.device, save=args.save)

    print(f"\nPredictions: {len(predictions)} detections")
    for pred in predictions:
        print(f"  {pred['class']}: {pred['confidence']:.3f} "
              f"bbox=({pred['bbox']['x1']:.0f}, {pred['bbox']['y1']:.0f}, "
              f"{pred['bbox']['x2']:.0f}, {pred['bbox']['y2']:.0f})")

    json_path = os.path.join(os.path.dirname(MODEL_PATH), "predictions.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    print(f"\nPredictions saved to {json_path}")


if __name__ == "__main__":
    main()
