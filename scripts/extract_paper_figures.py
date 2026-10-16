"""Render selected illustrative panels from the authors' official ICML 2022 slides.

Source: https://icml.cc/media/icml-2022/Slides/17920.pdf
The slide index is zero-based. Runs only for repository asset creation, never
as part of the default offline demo. Install pymupdf before running.
"""
from __future__ import annotations
import io
from pathlib import Path
from urllib.request import Request, urlopen
import fitz

ROOT = Path(__file__).resolve().parents[1]
URL = "https://icml.cc/media/icml-2022/Slides/17920.pdf"
# Select the visible panels and remove outer conference slide margins.
FIGURES = {
    "paper_metrics.png": (3, (.055, .18, .93, .88)),
    "paper_cid_ru.png": (6, (.065, .16, .93, .86)),
    "paper_architectures.png": (7, (.055, .16, .93, .85)),
    "paper_comparison.png": (9, (.055, .16, .93, .85)),
}


def main():
    request = Request(URL, headers={"User-Agent": "DeepInfo-research-asset/1.0"})
    with urlopen(request, timeout=45) as reply:
        pdf_bytes = reply.read()
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("Downloaded slide deck is not a PDF")
    pdf = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    assert len(pdf) >= 11, "expected 11 ICML slides"
    output = ROOT / "assets/figures"
    output.mkdir(parents=True, exist_ok=True)
    for file_name, (slide, coords) in FIGURES.items():
        page = pdf[slide]
        box = page.rect
        l, t, r, b = coords
        crop = fitz.Rect(box.width*l, box.height*t, box.width*r, box.height*b)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=crop, alpha=False)
        (output/file_name).write_bytes(pixmap.tobytes("png"))
        print(f"Wrote {file_name} ({pixmap.width} × {pixmap.height})")


if __name__ == "__main__":
    main()
