"""
Onion Quality Grading & Traceability Backend API
Unified REST Service integrating YOLOv8 Instance Segmentation, ArUco Metric Calibration,
Government Policy Rule Engine, and Audit Report Generation.
"""
from __future__ import annotations
import base64
import io
import json
import os
import sys
from pathlib import Path
from typing import Optional

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import torch

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml_backend.pipeline import process_image
from ml_backend.vision.grade import load_policy, GradingPolicy
from ml_backend.vision.calibrate import DEFAULT_MARKER_SIZE_MM

app = Flask(__name__, static_folder=None)
CORS(app, resources={r"/*": {"origins": "*"}})

REPORTS_DIR = REPO_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

VALID_IMAGES_DIR = REPO_ROOT / "data" / "real" / "images" / "valid"
SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"
FRONTEND_FILE = Path(__file__).resolve().parent / "frontend.html"

# Device identification
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "CPU"


@app.route("/", methods=["GET"])
def index():
    """Serve the primary interactive frontend UI."""
    if FRONTEND_FILE.is_file():
        return send_file(str(FRONTEND_FILE), mimetype="text/html")
    return jsonify({"error": "Frontend UI file not found"}), 404


@app.route("/health", methods=["GET"])
def health():
    """Health check and model telemetry."""
    policy = load_policy()
    return jsonify({
        "status": "online",
        "system": "Onion Quality Grading & Audit System (SIH)",
        "model": "YOLOv8s-seg (Instance Segmentation)",
        "model_accuracy": "93.6% Box mAP50 | 91.3% Mask mAP50",
        "device": f"{DEVICE_NAME} ({'CUDA GPU' if CUDA_AVAILABLE else 'CPU'})",
        "cuda_active": CUDA_AVAILABLE,
        "calibration_standard": f"ArUco 4x4_50 ({DEFAULT_MARKER_SIZE_MM} mm)",
        "active_policy": policy.to_dict(),
    })


@app.route("/policy", methods=["GET"])
def get_policy():
    """Retrieve the current active grading policy."""
    policy = load_policy()
    return jsonify(policy.to_dict())


@app.route("/policy", methods=["POST"])
def update_policy():
    """Update active grading policy thresholds."""
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    try:
        updated = GradingPolicy(**data)
        policy_path = REPO_ROOT / "shared" / "policies" / "default_policy.json"
        with open(policy_path, "w", encoding="utf-8") as f:
            json.dump(updated.to_dict(), f, indent=2)
        return jsonify({"status": "success", "policy": updated.to_dict()})
    except Exception as e:
        return jsonify({"error": f"Policy validation failed: {str(e)}"}), 400


@app.route("/predict", methods=["POST"])
@app.route("/grade", methods=["POST"])
def predict():
    """
    Run full grading pipeline on an uploaded image.
    Accepts multipart/form-data 'file' or JSON 'image' (base64 string).
    """
    img_bgr = None

    # Handle multipart upload
    if "file" in request.files:
        file = request.files["file"]
        if file.filename != "":
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Handle base64 JSON payload
    if img_bgr is None and request.is_json:
        data = request.get_json(silent=True) or {}
        b64_str = data.get("image", "")
        if b64_str:
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            try:
                img_bytes = base64.b64decode(b64_str)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                return jsonify({"error": f"Base64 decode failed: {str(e)}"}), 400

    if img_bgr is None:
        return jsonify({"error": "No valid image provided. Supply 'file' form field or 'image' base64."}), 400

    # Extract optional metadata
    batch_id = request.form.get("batch_id") or (request.json.get("batch_id") if request.is_json else None) or "BATCH-MANDI-001"
    farmer_id = request.form.get("farmer_id") or (request.json.get("farmer_id") if request.is_json else None) or "FARMER-IND-401"
    officer_id = request.form.get("officer_id") or (request.json.get("officer_id") if request.is_json else None) or "OFFICER-01"
    centre = request.form.get("centre") or (request.json.get("centre") if request.is_json else None) or "Lasalgaon Mandi Centre"

    gps_lat = request.form.get("gps_lat") or (request.json.get("gps_lat") if request.is_json else None)
    gps_lon = request.form.get("gps_lon") or (request.json.get("gps_lon") if request.is_json else None)
    gps = (float(gps_lat), float(gps_lon)) if gps_lat and gps_lon else (20.1462, 74.2285)

    conf_thresh = float(request.form.get("conf_threshold") or (request.json.get("conf_threshold") if request.is_json else 0.35))

    # Execute Full Pipeline
    try:
        res = process_image(
            image_input=img_bgr,
            batch_id=batch_id,
            farmer_id=farmer_id,
            officer_id=officer_id,
            centre=centre,
            gps=gps,
            conf_threshold=conf_thresh,
            generate_artifacts=True,
            output_dir=str(REPORTS_DIR),
        )

        # Encode annotated image to JPEG base64 for direct browser rendering
        annotated_bgr = res["annotated_image"]
        success, buffer = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        annotated_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}" if success else None

        # Build report URLs
        report_id = res.get("report_id")
        report_urls = None
        if report_id:
            report_urls = {
                "html": f"/reports/{report_id}.html",
                "pdf": f"/reports/{report_id}.pdf",
                "annotated_png": f"/reports/{report_id}_annotated.png",
                "qr_png": f"/reports/{report_id}_qr.png",
            }

        return jsonify({
            "status": res["status"],
            "is_onion_frame": res["is_onion_frame"],
            "summary": res["summary"],
            "calibration": res["calibration"],
            "predictions": res["onions"],
            "count": len(res["onions"]),
            "annotated_image": annotated_b64,
            "report_id": report_id,
            "report_urls": report_urls,
            "filter_summary": res["filter_summary"],
        })

    except Exception as err:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Grading pipeline failed: {str(err)}"}), 500


@app.route("/sample-images", methods=["GET"])
def list_sample_images():
    """List curated sample validation images for one-click testing."""
    samples = [
        {
            "id": "sample_prime_grade_a",
            "title": "Grade A (Prime) Onion",
            "category": "Grade A",
            "filename": "image_313_jpg.rf.cf3f0bfae6ff67add065c0827591da0e.jpg",
            "description": "Standard prime quality commercial onion meeting optimal market bounds.",
            "url": "/sample-images/image_313_jpg.rf.cf3f0bfae6ff67add065c0827591da0e.jpg",
        },
        {
            "id": "sample_grade_urs",
            "title": "Grade URS (Slight Defect)",
            "category": "Grade URS",
            "filename": "Class-1-Extra-Large-7-5-Slight-Shape-Defect-18-_jpg.rf.0df2ae87ee677c7b73c67af07a582247.jpg",
            "description": "Onion with slight surface skin peeling or slight shape irregularity.",
            "url": "/sample-images/Class-1-Extra-Large-7-5-Slight-Shape-Defect-18-_jpg.rf.0df2ae87ee677c7b73c67af07a582247.jpg",
        },
        {
            "id": "sample_reject",
            "title": "Reject (Severe Defect)",
            "category": "Reject",
            "filename": "image_477_jpg.rf.e121e49c0b7964e699e3f9fc497ced2f.jpg",
            "description": "Onion with severe rot, deep damage, or rejected quality defects.",
            "url": "/sample-images/image_477_jpg.rf.e121e49c0b7964e699e3f9fc497ced2f.jpg",
        },
        {
            "id": "sample_calibrated_tray",
            "title": "Calibrated Multi-Onion Tray",
            "category": "Multi-Onion Tray",
            "filename": "composite_tray.jpg",
            "description": "Multi-onion procurement tray with 25.0 mm ArUco calibration marker.",
            "url": "/sample-images/composite_tray.jpg",
        },
    ]
    return jsonify(samples)


@app.route("/sample-images/<filename>", methods=["GET"])
def get_sample_image(filename):
    """Serve sample image from validation dataset or synthetic directory."""
    valid_path = VALID_IMAGES_DIR / filename
    if valid_path.is_file():
        return send_file(str(valid_path), mimetype="image/jpeg")

    synth_path = SYNTHETIC_DIR / filename
    if synth_path.is_file():
        return send_file(str(synth_path), mimetype="image/jpeg")

    return jsonify({"error": f"Sample image {filename} not found"}), 404


@app.route("/reports/<report_id>/pdf", methods=["GET"])
def download_pdf(report_id):
    """Download certified PDF inspection certificate."""
    pdf_path = REPORTS_DIR / f"{report_id}.pdf"
    if pdf_path.is_file():
        return send_file(str(pdf_path), as_attachment=True, download_name=f"{report_id}.pdf", mimetype="application/pdf")
    return jsonify({"error": f"Report PDF {report_id} not found"}), 404


@app.route("/reports/<report_id>/html", methods=["GET"])
def view_html(report_id):
    """View digital HTML inspection certificate."""
    html_path = REPORTS_DIR / f"{report_id}.html"
    if html_path.is_file():
        return send_file(str(html_path), mimetype="text/html")
    return jsonify({"error": f"Report HTML {report_id} not found"}), 404


@app.route("/reports/<path:filename>", methods=["GET"])
def get_report_asset(filename):
    """Serve report static assets (annotated images, QR codes)."""
    file_path = REPORTS_DIR / filename
    if file_path.is_file():
        return send_file(str(file_path))
    return jsonify({"error": f"Report asset {filename} not found"}), 404


if __name__ == "__main__":
    print(f"Starting Onion Grading Backend on port 5000 (Device: {DEVICE_NAME})...")
    app.run(host="0.0.0.0", port=5000, debug=False)
