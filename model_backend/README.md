# Onion Grading Backend

YOLOv8 segmentation model for onion quality grading.

## Structure
- `best.pt` - Trained model weights (best checkpoint)
- `last.pt` - Final model weights (last epoch)
- `onion.yaml` - Dataset config (3 classes: grade_a, grade_urs, reject)
- `inference.py` - Command-line inference tool
- `api.py` - Flask REST API server

## Classes
| ID | Name | Description |
|----|------|-------------|
| 0 | grade_a | Grade A onions (Extra Class + Class 1) |
| 1 | grade_urs | Under-ripe/Slight defects (Class 2) |
| 2 | reject | Rejected onions |

## Install
```bash
pip install -r requirements.txt
```

## CLI Usage
```bash
# Single image
python inference.py --source image.jpg --conf 0.5

# Folder of images
python inference.py --source path/to/images/ --conf 0.5 --save
```

## API Usage
```bash
python api.py
```

```bash
# Predict
curl -X POST http://localhost:5000/predict \
  -F "file=@onion.jpg"

# Health
curl http://localhost:5000/health
```

## Requirements
- Python 3.9+
- CUDA GPU recommended (RTX 5050 or similar)
- ultralytics, torch, opencv-python, flask, Pillow