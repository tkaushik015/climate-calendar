"""Extract raw text from PAU Package of Practices PDFs.

Outputs one .txt file per PDF in finetune/sources/extracted/.
We'll feed sections of these into Claude tomorrow to synthesize Q&A pairs.
"""

from pathlib import Path

from pypdf import PdfReader


def extract(pdf_path: Path, out_path: Path) -> None:
    reader = PdfReader(str(pdf_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for i, page in enumerate(reader.pages, start=1):
            f.write(f"\n\n=== PAGE {i} ===\n\n")
            f.write(page.extract_text() or "")
    print(f"✓ Extracted {len(reader.pages)} pages from {pdf_path.name}")
    print(f"  Output: {out_path} ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    sources_dir = Path(__file__).parent / "sources"
    extracted_dir = sources_dir / "extracted"

    for pdf in sources_dir.glob("*.pdf"):
        out = extracted_dir / (pdf.stem + ".txt")
        extract(pdf, out)
