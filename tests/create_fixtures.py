"""Generate minimal test fixture files with valid magic bytes."""
import os

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def create_fixtures():
    """Create minimal binary files with correct magic signatures."""
    os.makedirs(FIXTURES_DIR, exist_ok=True)

    # Minimal PDF (valid header)
    with open(os.path.join(FIXTURES_DIR, "sample.pdf"), "wb") as f:
        f.write(b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
                b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
                b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
                b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
                b"0000000058 00000 n \n0000000115 00000 n \n"
                b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n193\n%%EOF")

    # PNG (8-byte signature)
    with open(os.path.join(FIXTURES_DIR, "sample.png"), "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

    # JPEG (FFD8FF signature)
    with open(os.path.join(FIXTURES_DIR, "sample.jpg"), "wb") as f:
        f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 50)

    # TIFF little-endian (II + 42)
    with open(os.path.join(FIXTURES_DIR, "sample.tif"), "wb") as f:
        f.write(b"II*\x00" + b"\x00" * 50)

    # BMP
    with open(os.path.join(FIXTURES_DIR, "sample.bmp"), "wb") as f:
        f.write(b"BM" + b"\x00" * 50)

    # GIF89a
    with open(os.path.join(FIXTURES_DIR, "sample.gif"), "wb") as f:
        f.write(b"GIF89a" + b"\x00" * 50)

    # WebP (RIFF + WEBP)
    with open(os.path.join(FIXTURES_DIR, "sample.webp"), "wb") as f:
        f.write(b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 50)

    # ICO
    with open(os.path.join(FIXTURES_DIR, "sample.ico"), "wb") as f:
        f.write(b"\x00\x00\x01\x00" + b"\x00" * 50)

    # PNM (P6 — binary PPM)
    with open(os.path.join(FIXTURES_DIR, "sample.ppm"), "wb") as f:
        f.write(b"P6\n2 2\n255\n" + b"\xff\x00\x00" * 4)

    # SVG
    with open(os.path.join(FIXTURES_DIR, "sample.svg"), "wb") as f:
        f.write(b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"></svg>')

    # JPEG 2000 (jp2)
    with open(os.path.join(FIXTURES_DIR, "sample.jp2"), "wb") as f:
        f.write(b"\x00\x00\x00\x0cjP  \r\n\x87\n" + b"\x00" * 50)

    # Unsupported extension
    with open(os.path.join(FIXTURES_DIR, "data.csv"), "w") as f:
        f.write("col1,col2\na,b\n")

    # Empty file
    with open(os.path.join(FIXTURES_DIR, "empty.pdf"), "wb") as f:
        pass

    # Mismatch: .pdf extension but PNG magic bytes
    with open(os.path.join(FIXTURES_DIR, "mismatch.pdf"), "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

    # Phase 2 extension (not yet enabled)
    with open(os.path.join(FIXTURES_DIR, "photo.heic"), "wb") as f:
        f.write(b"\x00" * 50)


if __name__ == "__main__":
    create_fixtures()
    print(f"Fixtures created in {FIXTURES_DIR}")
