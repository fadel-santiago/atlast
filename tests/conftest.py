from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
from typing import Sequence

import pytest


class DeterministicTestEmbedder:
    """Small dependency-free embedder used to exercise storage and retrieval."""

    dimension = 16
    name = "test/hash-embedding-v1"

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimension
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.sha256(word.encode("utf-8")).digest()
                slot = int.from_bytes(digest[:2], "big") % self.dimension
                vector[slot] += 1.0 if digest[2] % 2 else -1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


@pytest.fixture
def embedder() -> DeterministicTestEmbedder:
    return DeterministicTestEmbedder()


def write_synthetic_pdf(path: Path, page_texts: Sequence[str]) -> Path:
    """Write a minimal valid PDF with extractable Helvetica text."""

    page_count = len(page_texts)
    font_object = 3 + page_count * 2
    kids = " ".join(f"{3 + index} 0 R" for index in range(page_count))
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode(),
    ]

    for index in range(page_count):
        content_object = 3 + page_count + index
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_object} 0 R >> >> "
                f"/Contents {content_object} 0 R >>"
            ).encode()
        )

    for text in page_texts:
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode()
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")

    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_number} 0 obj\n".encode())
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    path.write_bytes(pdf)
    return path


@pytest.fixture
def synthetic_pdf(tmp_path: Path) -> Path:
    return write_synthetic_pdf(
        tmp_path / "sample.pdf",
        ["Alpha appears on the first page.", "Beta appears on the second page."],
    )
