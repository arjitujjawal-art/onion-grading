"""Prepare the onion grading dataset for YOLOv8 training.
- Merges part-001 (train+test) and part-002 (train+valid) into a unified dataset
- Converts COCO segmentation JSON to YOLOv8 txt format
- Maps categories: Extra Class + Class 1 -> GRADE_A(0), Class 2 -> GRADE_URS(1), Reject -> REJECTED(2)
- Splits into train (80%) / valid (20%)
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
from collections import Counter

# Source: cloned dataset repo
SRC = Path("Onion-grading-deeplearning-SIH-/dataset")

# Destination: organized as data/real/
DST = Path("data/real")
IMG_DST = DST / "images"
LBL_DST = DST / "labels"

# Category mapping: original COCO category_id -> YOLO class id
# Class 1 (id=1), Extra Class (id=3) -> GRADE_A (0)
# Class 2 (id=2) -> GRADE_URS (1)
# Reject (id=4) -> REJECTED (2)
# "Onions" (id=0) is the parent category, will be ignored
CAT_MAP = {1: 0, 3: 0, 2: 1, 4: 2}  # 0=grade_a, 1=grade_urs, 2=reject
CLS_NAMES = {0: "grade_a", 1: "grade_urs", 2: "reject"}

def load_coco_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

def convert_annotation(coco: dict, image_dir: Path, label_dir: Path, split: str) -> list[str]:
    """Convert COCO annotations to YOLOv8 txt files. Returns list of (split, filename) pairs."""
    img_entries = {img["id"]: img for img in coco["images"]}
    label_entries = {}  # image_id -> list of annotation strings

    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in CAT_MAP:
            continue
        cls_id = CAT_MAP[cat_id]

        # bbox is [x, y, w, h] in abs pixels
        x, y, w, h = ann["bbox"]
        seg = ann.get("segmentation")

        # Normalize bbox to YOLO format
        img = img_entries[ann["image_id"]]
        iw, ih = img["width"], img["height"]
        cx, cy = x + w / 2, y + h / 2
        line = f"{cls_id} {cx/iw:.6f} {cy/ih:.6f} {w/iw:.6f} {h/ih:.6f}"

        # If segmentation exists and is a polygon, include it
        if seg and isinstance(seg, list):
            for poly in seg:
                pts = poly
                norm_pts = []
                for i in range(0, len(pts), 2):
                    px, py = pts[i] / iw, pts[i + 1] / ih
                    norm_pts.extend([f"{px:.6f}", f"{py:.6f}"])
                line += " " + " ".join(norm_pts)
                break  # Only first polygon (single onion per annotation)

        label_entries.setdefault(ann["image_id"], []).append(line)

    # Write label files
    results = []
    for img_id, img in img_entries.items():
        fname = img["file_name"]
        src_img = image_dir / fname
        dst_img = IMG_DST / split / fname
        label_name = fname.rsplit(".", 1)[0] + ".txt"
        dst_lbl = LBL_DST / split / label_name

        if not src_img.exists():
            continue

        (IMG_DST / split).mkdir(parents=True, exist_ok=True)
        (LBL_DST / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_img, dst_img)

        labels = label_entries.get(img_id, [])
        dst_lbl.write_text("\n".join(labels) + ("\n" if labels else ""))

        results.append(f"{split} {fname}")

    return results

def main():
    if IMG_DST.exists():
        shutil.rmtree(IMG_DST)
    (IMG_DST / "train").mkdir(parents=True, exist_ok=True)
    (IMG_DST / "valid").mkdir(parents=True, exist_ok=True)

    all_entries = []
    cat_counter = Counter()

    # --- Part 001: onion-grading-coco-segmentation (train + test) ---
    p1_base = SRC / "part-001" / "onion-grading-coco-segmentation"
    coco_train = load_coco_json(p1_base / "train" / "_annotations.coco.json")
    all_entries += convert_annotation(coco_train, p1_base / "train", LBL_DST, "train")
    for ann in coco_train["annotations"]:
        if ann["category_id"] in CAT_MAP:
            cat_counter[CAT_MAP[ann["category_id"]]] += 1

    coco_test = load_coco_json(p1_base / "test" / "_annotations.coco.json")
    all_entries += convert_annotation(coco_test, p1_base / "test", LBL_DST, "valid")
    for ann in coco_test["annotations"]:
        if ann["category_id"] in CAT_MAP:
            cat_counter[CAT_MAP[ann["category_id"]]] += 1

    # --- Part 002: onion-grading-coco-segmentation (train + valid) ---
    p2_base = SRC / "part-002" / "onion-grading-coco-segmentation"
    # Part-002 train has images but no annotations JSON — only valid has annotations
    coco_p2_valid = load_coco_json(p2_base / "valid" / "_annotations.coco.json")
    all_entries += convert_annotation(coco_p2_valid, p2_base / "valid", LBL_DST, "valid")
    for ann in coco_p2_valid["annotations"]:
        if ann["category_id"] in CAT_MAP:
            cat_counter[CAT_MAP[ann["category_id"]]] += 1

    # Write dataset YAML for YOLOv8
    train_count = len(list((IMG_DST / "train").glob("*.jpg")))
    valid_count = len(list((IMG_DST / "valid").glob("*.jpg")))
    yaml_path = DST / "onion.yaml"
    yaml_content = f"""# Onion grading dataset (YOLOv8 segmentation format)
path: ../data/real
train: images/train
val: images/valid

nc: 3
names: ['grade_a', 'grade_urs', 'reject']
"""
    yaml_path.write_text(yaml_content)

    print(f"Dataset prepared: {yaml_path}")
    print(f"  Train images: {train_count}")
    print(f"  Valid images: {valid_count}")
    print(f"  Annotation distribution: {dict((CLS_NAMES[k], v) for k, v in cat_counter.items())}")

if __name__ == "__main__":
    main()
