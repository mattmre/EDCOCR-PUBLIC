# File Format Support & Limitations

## (15 formats + PDF)

| Format | Extension | Magic Bytes | Notes |
|---|---|---|---|
| PDF | .pdf | %PDF- | Full support, multi-page |
| TIFF | .tif, .tiff | `II\x2a\x00` or `MM\x00\x2a` | Multi-frame support (all pages extracted) |
| JPEG | .jpg, .jpeg | \xff\xd8\xff | Standard support |
| PNG | .png | \x89PNG | Standard support |
| BMP | .bmp | BM | Standard support |
| GIF | .gif | GIF87a or GIF89a | Multi-frame support |
| WebP | .webp | RIFF...WEBP | Standard support |
| JPEG 2000 | .jp2, .jpx | \x00\x00\x00\x0cjP | Via PyMuPDF fallback |
| PNM/PBM/PGM/PPM | .pnm, .pbm, .pgm, .ppm | P1-P6 | Netpbm formats |
| PCX | .pcx | — | Via PIL |
| ICO | .ico | \x00\x00\x01\x00 | Windows icon format |
| SVG | .svg, .svgz | <svg | Vector, converted to raster at DPI |

## File Classification Process
1. Extension check against allowlists (PDF_EXTENSIONS, PHASE1_IMAGE_EXTENSIONS)
2. Magic byte signature verification (detect_magic_family)
3. Cross-validation: if extension says "pdf" but magic says "image" → REJECTED (signature mismatch)
4. If no magic match found → accepted by extension fallback with warning logged

## (Not Yet Enabled)

| Format | Extension | Status | Blocker |
|---|---|---|---|
| HEIC/HEIF | .heic, .heif | Planned | Requires pillow-heif or imagemagick |
| AVIF | .avif | Planned | Requires pillow-avif-plugin |
| JPEG XL | .jxl | Planned | Requires jxlpy or imagemagick |
| JPEG XR | .jxr | Planned | Limited Python library support |
| DCX | .dcx | Planned | Multi-page PCX, needs custom reader |
| XPS | .xps | Planned | Microsoft format, needs python-xps or conversion |

## Output Format Handling
- **Non-PDF sources**: Output filename includes extension token to prevent collisions
  - Example: `photo.tiff` → `photo__tiff.pdf` and `photo__tiff.txt`
- **Multi-frame images**: Each frame treated as a separate page in output PDF
- **SVG**: Rasterized at configured DPI (default 300) before OCR

## Known Format-Specific Issues
- **Encrypted PDFs**: Not supported — will fail at page count step
- **PDF/A**: Supported but OCR output is not PDF/A compliant
- **Password-protected PDFs**: Not supported
- **Corrupted files**: Detected during extraction, logged to failures.csv, skipped gracefully
- **Very large images** (>100MP): May cause memory issues during PIL conversion
- **16-bit TIFF**: Automatically converted to 8-bit RGB
