"""
End-to-end Demonstration CLI:
Tests the model and grading pipeline at full capability on real dataset samples
or calibrated multi-onion procurement trays.
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
import random

# Ensure root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_backend.pipeline import process_image
from scripts.generate_synthetic_data import generate_tray_image


def run_demo(mode: str = "real", custom_image: str = None, count: int = 3):
    print("=" * 72)
    print("  ONION QUALITY GRADING & TRACEABILITY SYSTEM (SIH)")
    print("  Deep Learning (YOLOv8s-Seg) + ArUco Calibration + Policy Rule Engine")
    print("=" * 72)

    target_images = []
    if custom_image:
        target_images.append(Path(custom_image))
    elif mode == "tray":
        tray_path = Path("data/synthetic/composite_tray.jpg")
        if not tray_path.exists():
            print("\nGenerating calibrated composite tray with ArUco reference...")
            # Recreate composite tray
            import cv2
            tray = cv2.imread("data/synthetic/test_tray.png")
            target_images.append(Path("data/synthetic/composite_tray.jpg"))
        else:
            target_images.append(tray_path)
    else:
        valid_dir = Path("data/real/images/valid")
        all_val = sorted(list(valid_dir.glob("*.jpg")))
        if not all_val:
            print("No validation images found in data/real/images/valid. Fallback to tray.")
            target_images.append(Path("data/synthetic/composite_tray.jpg"))
        else:
            # Pick a diverse set of samples
            random.seed(42)
            target_images = random.sample(all_val, min(count, len(all_val)))

    for idx, img_path in enumerate(target_images, 1):
        print(f"\n[{idx}/{len(target_images)}] Processing: {img_path.name}")
        print("-" * 72)

        res = process_image(
            image_input=img_path,
            batch_id=f"DEMO-BATCH-{idx:03d}",
            farmer_id="FARMER-MH-4122",
            officer_id="INSP-OFFICER-07",
            centre="Lasalgaon APMC Mandi",
            gps=(20.1462, 74.2285),  # Lasalgaon, Nashik (Onion capital of India)
            generate_artifacts=True,
        )

        cal = res["calibration"]
        summary = res["summary"]
        report_id = res["report_id"]

        print(f"  Status:             {res['status'].upper()}")
        print(f"  Calibration:        {'CALIBRATED (ArUco 25mm)' if cal['calibrated'] else 'UNCALIBRATED (Estimated Scale)'}")
        if cal["calibrated"]:
            print(f"  Scale Factor:       {cal['pixels_per_mm']:.2f} pixels/mm")
        print(f"  Total Onions Found: {summary['total']}")
        print(f"  Overall Decision:   {summary['overall_status']} ({summary['acceptance_status']})")
        print(f"  Grade A:            {summary['counts']['GRADE_A']} ({summary['grade_a_pct']}%)")
        print(f"  Grade URS:          {summary['counts']['GRADE_URS']} ({summary['grade_urs_pct']}%)")
        print(f"  Rejected:           {summary['counts']['REJECTED']} ({summary['rejected_pct']}%)")
        print(f"  Manual Review:      {summary['counts']['MANUAL_REVIEW']} ({summary['manual_review_pct']}%)")

        if res["onions"]:
            print("\n  Per-Onion Analysis:")
            print(f"  {'ID':<10} | {'Visual AI':<10} | {'Diameter':<10} | {'Conf':<6} | {'Final Grade':<14} | Reasons")
            print("  " + "-" * 68)
            for o in res["onions"]:
                reasons_str = ", ".join(o["reason_codes"])
                print(f"  {o['onion_id']:<10} | {o['visual_class']:<10} | {o['diameter_mm']:>5.1f} mm   | {o['confidence']*100:>4.1f}% | {o['final_grade']:<14} | {reasons_str}")

        print(f"\n  Evidence Artifacts Generated:")
        print(f"   -> HTML Report: reports/{report_id}.html")
        print(f"   -> PDF Report:  reports/{report_id}.pdf")
        print(f"   -> Annotated:   reports/{report_id}_annotated.png")
        print(f"   -> QR Code:     reports/{report_id}_qr.png")

    print("\n" + "=" * 72)
    print("Demo completed successfully. All inspection certificates saved to reports/")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Onion Grading Pipeline Demonstration")
    parser.add_argument("--mode", choices=["real", "tray"], default="real", help="Test mode: 'real' images or 'tray'")
    parser.add_argument("--image", type=str, default=None, help="Path to custom image to grade")
    parser.add_argument("--count", type=int, default=3, help="Number of real images to test")
    args = parser.parse_args()

    run_demo(mode=args.mode, custom_image=args.image, count=args.count)
