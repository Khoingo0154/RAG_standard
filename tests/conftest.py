import io
import random
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_sample_pdf(num_pages: int = 3) -> bytes:
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for i in range(num_pages):
        c.setFont("Helvetica", 12)
        c.drawString(100, 700, f"Page {i+1} of test document.")
        c.drawString(100, 680, f"This is sample content for page {i+1}.")
        c.drawString(100, 660, "Testing chunking and embedding pipeline.")
        c.drawString(100, 640, "Additional text to ensure enough content for valid chunks.")
        c.drawString(100, 620, "The chunking system should process all of this data correctly.")
        c.drawString(100, 600, "We need more text to meet the minimum chunk requirements.")
        c.drawString(100, 580, "This should be enough characters for the validation checks.")
        if i == 0:
            c.drawString(100, 560, "Trang 1")
            c.drawString(100, 540, "1/3")
        c.showPage()
    c.save()
    return buf.getvalue()
