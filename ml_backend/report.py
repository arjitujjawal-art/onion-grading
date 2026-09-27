"""Generate evidence-backed digital reports: HTML + QR + PDF."""
from __future__ import annotations
import json
import qrcode
import io
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image
import numpy as np

from ml_backend.vision.grade import grade_batch, OnionResult

def generate_qr_code(report_id: str) -> Image.Image:
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(report_id)
    qr.make(fit=True)
    pil_img = qr.make_image(fill_color="black", back_color="white")
    # Convert qrcode's PilImage wrapper to a standard PIL Image
    return Image.fromarray(np.array(pil_img.get_image()))

def generate_report(
    batch_id: str,
    farmer_id: str,
    officer_id: str,
    centre: str,
    policy: dict,
    results: list[OnionResult],
    original_image: np.ndarray,
    annotated_image: np.ndarray,
    gps: Optional[Tuple[float, float]] = None,
    output_dir: str = "reports",
) -> str:
    """Generate HTML + PDF report with QR code and annotated images."""
    Path(output_dir).mkdir(exist_ok=True)
    report_id = f"ONION-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    batch_stats = grade_batch(results)

    # Convert numpy images to PIL for embedding
    orig_pil = Image.fromarray(original_image) if isinstance(original_image, np.ndarray) else original_image
    annot_pil = Image.fromarray(annotated_image) if isinstance(annotated_image, np.ndarray) else annotated_image

    # Generate QR code
    qr_img = generate_qr_code(report_id)

    # Save annotated image
    annot_path = Path(output_dir) / f"{report_id}_annotated.png"
    annot_pil.save(annot_path, format="PNG")

    orig_path = Path(output_dir) / f"{report_id}_original.png"
    orig_pil.save(orig_path, format="PNG")

    qr_path = Path(output_dir) / f"{report_id}_qr.png"
    qr_img.save(qr_path, format="PNG")

    # Generate HTML report
    html = _generate_html(
        report_id, batch_id, farmer_id, officer_id, centre,
        policy, batch_stats, results, gps, annot_path, orig_path, qr_path
    )
    html_path = Path(output_dir) / f"{report_id}.html"
    html_path.write_text(html)

    # Generate PDF
    pdf_path = Path(output_dir) / f"{report_id}.pdf"
    _generate_pdf(pdf_path, report_id, html, annot_pil, orig_pil, qr_img, batch_stats)

    return str(report_id)

def _generate_html(report_id: str, batch_id: str, farmer_id: str,
                   officer_id: str, centre: str, policy: dict,
                   batch_stats: dict, results: list[OnionResult],
                   gps: Optional[Tuple[float, float]], annot_path: Path,
                   orig_path: Path, qr_path: Path) -> str:
    gps_str = f"{gps[0]:.6f}, {gps[1]:.6f}" if gps else "N/A"

    # Per-onion table rows
    rows = []
    for r in results:
        defects_str = ", ".join([k for k, v in r.defects.__dict__.items() if v]) or "none"
        reasons = ", ".join(r.reason_codes) if r.reason_codes else "—"
        rows.append(f"""
            <tr>
              <td>{r.onion_id}</td>
              <td>{r.diameter_mm:.1f} mm</td>
              <td>{defects_str}</td>
              <td>{r.confidence:.2f}</td>
              <td>{r.final_grade}</td>
              <td>{reasons}</td>
            </tr>
        """)

    return f"""
    <!DOCTYPE html>
    <html>
    <head><title>Onion Grading Report — {report_id}</title>
    <style>
      body {{ font-family: sans-serif; margin: 2em; }}
      h1 {{ color: #2e7d32; }}
      table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
      th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
      th {{ background: #f0f0f0; }}
      img {{ max-width: 100%; margin: 0.5em 0; }}
      .summary {{ background: #f9f9f9; padding: 1em; border-radius: 8px; }}
    </style>
    </head>
    <body>
      <h1>Onion Quality Grading Report</h1>
      <div class="summary">
        <p><strong>Report ID:</strong> {report_id}</p>
        <p><strong>Batch ID:</strong> {batch_id} | <strong>Farmer ID:</strong> {farmer_id}</p>
        <p><strong>Procurement Centre:</strong> {centre} | <strong>Officer ID:</strong> {officer_id}</p>
        <p><strong>Date/Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>GPS:</strong> {gps_str}</p>
        <img src="{qr_path.name}" alt="QR Code" style="width: 100px; float: right;">
      </div>

      <h2>Policy</h2>
      <pre>{json.dumps(policy, indent=2)}</pre>

      <h2>Batch Summary</h2>
      <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Onions</td><td>{batch_stats['total']}</td></tr>
        <tr><td>Grade A Count</td><td>{batch_stats['counts'].get('GRADE_A', 0)}</td></tr>
        <tr><td>Grade A %</td><td>{batch_stats['grade_a_pct']}%</td></tr>
        <tr><td>Grade URS Count</td><td>{batch_stats['counts'].get('GRADE_URS', 0)}</td></tr>
        <tr><td>Grade URS %</td><td>{batch_stats['grade_urs_pct']}%</td></tr>
        <tr><td>Rejected Count</td><td>{batch_stats['counts'].get('REJECTED', 0)}</td></tr>
        <tr><td>Rejected %</td><td>{batch_stats['rejected_pct']}%</td></tr>
        <tr><td>Manual Review Count</td><td>{batch_stats['counts'].get('MANUAL_REVIEW', 0)}</td></tr>
        <tr><td>Manual Review %</td><td>{batch_stats['manual_review_pct']}%</td></tr>
      </table>

      <h2>Original Image</h2>
      <img src="{orig_path.name}" alt="Original">

      <h2>Annotated Image</h2>
      <img src="{annot_path.name}" alt="Annotated">

      <h2>Per-Onion Results</h2>
      <table>
        <tr><th>ID</th><th>Diameter</th><th>Defects</th><th>Confidence</th><th>Grade</th><th>Reasons</th></tr>
        {''.join(rows)}
      </table>
    </body>
    </html>
    """

def _generate_pdf(path, report_id, html, annotated_pil, original_pil, qr_img, batch_stats):
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter

    # Page 1
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, height - 1 * inch, "Onion Quality Grading Report")

    c.setFont("Helvetica", 10)
    c.drawString(1 * inch, height - 1.4 * inch, f"Report ID: {report_id}")
    c.drawString(1 * inch, height - 1.6 * inch, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Summary boxes
    y = height - 2.5 * inch
    c.setFont("Helvetica-Bold", 12)
    metrics = [
        ("Total Onions", str(batch_stats["total"])),
        ("Grade A %", f"{batch_stats['grade_a_pct']}%"),
        ("Grade URS %", f"{batch_stats['grade_urs_pct']}%"),
        ("Rejected %", f"{batch_stats['rejected_pct']}%"),
        ("Manual Review %", f"{batch_stats['manual_review_pct']}%"),
    ]
    for label, value in metrics:
        c.rect(1 * inch, y - 0.4 * inch, 2.5 * inch, 0.5 * inch, fill=1)
        c.setFillColorRGB(1, 1, 1)
        c.drawString(1.2 * inch, y - 0.25 * inch, f"{label}: {value}")
        c.setFillColorRGB(0, 0, 0)
        y -= 0.6 * inch

    # Annotated image
    img_width = 6 * inch
    img_height = 4.5 * inch
    c.drawImage(ImageReader(annotated_pil), 1 * inch, y - img_height - 0.5 * inch, width=img_width, height=img_height, preserveAspectRatio=True, mask='auto')

    # QR code
    qr_size = 1.5 * inch
    c.drawImage(ImageReader(qr_img), width - 2.5 * inch, y - img_height - 0.5 * inch, width=qr_size, height=qr_size, mask='auto')

    c.save()
