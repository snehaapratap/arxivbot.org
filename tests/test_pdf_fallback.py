"""Tests for indexing PDF-only arXiv submissions. No network, no LLM."""

import asyncio
import re
import gzip
import io
import tarfile

import pytest

from arxivbot.services import paper_service

PAPER_ID = "2609.00100v1"


def make_pdf(text: str | None) -> bytes:
    """Build a minimal one-page PDF with a Helvetica text stream (or no text)."""
    if text is None:
        stream = b""
    else:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_pos,
    )
    return bytes(out)


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    paper_service._indexing_status.clear()
    paper_service._content_cache.clear()

    async def noop_upsert(paper):
        return None

    monkeypatch.setattr(paper_service.db, "upsert_paper", noop_upsert)
    yield
    paper_service._indexing_status.clear()
    paper_service._content_cache.clear()


def serve(monkeypatch, body: bytes):
    async def fake_get(url, **kwargs):
        return 200, body

    monkeypatch.setattr(paper_service, "_arxiv_get", fake_get)


def fetch():
    return asyncio.run(paper_service.fetch_paper_content(PAPER_ID))


def status():
    return asyncio.run(paper_service.get_indexing_status(PAPER_ID))


def test_extractor_returns_text():
    text = paper_service._extract_text_from_pdf(make_pdf("Quantum widgets are great"))
    assert "Quantum widgets are great" in text


def test_pdf_source_is_indexed(monkeypatch):
    serve(monkeypatch, make_pdf("Quantum widgets are great"))
    content = fetch()
    assert "Quantum widgets are great" in content
    assert status()["status"] == "complete"


def test_gzipped_pdf_source_is_indexed(monkeypatch):
    serve(monkeypatch, gzip.compress(make_pdf("Gzipped widgets")))
    assert "Gzipped widgets" in fetch()


def test_textless_pdf_gives_scanned_error(monkeypatch):
    serve(monkeypatch, make_pdf(None))
    with pytest.raises(paper_service.PaperFetchError, match="scanned PDF"):
        fetch()
    st = status()
    assert st["status"] == "error"
    assert st["error"] == paper_service.SCANNED_PDF_MESSAGE


def test_garbage_pdf_gives_friendly_error(monkeypatch):
    serve(monkeypatch, b"%PDF-1.4\n\x00\x01garbage not a pdf at all\xff\xfe")
    with pytest.raises(paper_service.PaperFetchError):
        fetch()
    st = status()
    assert st["status"] == "error"
    assert st["error"] == paper_service.UNREADABLE_PDF_MESSAGE


def test_non_pdf_without_tex_gives_friendly_error(monkeypatch):
    serve(monkeypatch, b"just some random bytes, not TeX")
    with pytest.raises(paper_service.PaperFetchError, match=re.escape(paper_service.NO_TEX_SOURCE_MESSAGE)):
        fetch()
    assert status()["error"] == paper_service.NO_TEX_SOURCE_MESSAGE


def test_tar_source_unaffected(monkeypatch):
    tex = b"\\documentclass{article}\\begin{document}Tar hello\\end{document}"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo("main.tex")
        info.size = len(tex)
        tar.addfile(info, io.BytesIO(tex))
    serve(monkeypatch, buf.getvalue())
    content = fetch()
    assert "Tar hello" in content
    assert "main.tex" in content
    assert status()["status"] == "complete"
