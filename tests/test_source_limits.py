"""Tests for arXiv source size caps and off-loop extraction. No network, no LLM."""

import asyncio
import gzip
import io
import re
import tarfile

import pytest
from aioresponses import aioresponses

from arxivbot.services import paper_service
from arxivbot.utils.arxiv import parse_arxiv_id

PAPER_ID = "2301.00001"
SRC_URL = parse_arxiv_id(PAPER_ID).src_url
TEX = b"\\documentclass{article}\\begin{document}Hello\\end{document}"
TOO_LARGE = paper_service.SOURCE_TOO_LARGE_MESSAGE


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    paper_service._indexing_status.clear()
    paper_service._content_cache.clear()

    async def noop_upsert(paper):
        return None

    async def no_sleep(delay):
        return None

    monkeypatch.setattr(paper_service.db, "upsert_paper", noop_upsert)
    monkeypatch.setattr(paper_service, "_sleep", no_sleep)
    yield
    paper_service._indexing_status.clear()
    paper_service._content_cache.clear()


def fetch():
    return asyncio.run(paper_service.fetch_paper_content(PAPER_ID))


def request_count(m):
    return sum(len(v) for v in m.requests.values())


def make_tar_gz(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, body in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    return buf.getvalue()


def test_content_length_over_cap_fails_without_retry(monkeypatch):
    monkeypatch.setattr(paper_service, "MAX_SOURCE_BYTES", 1000)
    with aioresponses() as m:
        m.get(SRC_URL, status=200, body=TEX, headers={"Content-Length": "5000"}, repeat=True)
        with pytest.raises(paper_service.PaperFetchError, match=re.escape(TOO_LARGE)):
            fetch()
        assert request_count(m) == 1
    status = paper_service._indexing_status[PAPER_ID]
    assert status["status"] == "error"
    assert status["error"] == TOO_LARGE


def test_chunked_body_over_cap_fails_without_retry(monkeypatch):
    monkeypatch.setattr(paper_service, "MAX_SOURCE_BYTES", 1000)
    monkeypatch.setattr(paper_service, "_READ_CHUNK_BYTES", 256)
    with aioresponses() as m:
        m.get(SRC_URL, status=200, body=b"x" * 5000, repeat=True)
        with pytest.raises(paper_service.PaperFetchError, match=re.escape(TOO_LARGE)):
            fetch()
        assert request_count(m) == 1


def test_body_under_cap_is_read(monkeypatch):
    monkeypatch.setattr(paper_service, "MAX_SOURCE_BYTES", len(TEX))
    with aioresponses() as m:
        m.get(SRC_URL, status=200, body=TEX)
        assert "Hello" in fetch()


def test_gzip_bomb_rejected(monkeypatch):
    monkeypatch.setattr(paper_service, "MAX_DECOMPRESSED_BYTES", 1024 * 1024)
    # Non-tar filler: exercises the single-gzipped-file path (all-zero data parses as an empty tar)
    bomb = gzip.compress(b"A" * (20 * 1024 * 1024))
    with aioresponses() as m:
        m.get(SRC_URL, status=200, body=bomb)
        with pytest.raises(paper_service.PaperFetchError, match=re.escape(TOO_LARGE)):
            fetch()


def test_tar_bomb_rejected_before_full_decompression(monkeypatch):
    monkeypatch.setattr(paper_service, "MAX_DECOMPRESSED_BYTES", 1024 * 1024)
    reads = []
    real_stream = paper_service._gunzip_stream

    def counting_stream(data):
        reader = real_stream(data)
        real_read = reader.read

        def read(size=-1):
            chunk = real_read(size)
            reads.append(len(chunk))
            return chunk

        reader.read = read
        return reader

    monkeypatch.setattr(paper_service, "_gunzip_stream", counting_stream)
    data = make_tar_gz({"fig.bin": b"\0" * (20 * 1024 * 1024), "main.tex": TEX})
    with pytest.raises(paper_service.SourceTooLargeError):
        paper_service._extract_tex_from_tar(data)
    assert sum(reads) <= 2 * 1024 * 1024


def test_gzipped_pdf_bomb_rejected(monkeypatch):
    # Gzipped PDFs are capped at MAX_SOURCE_BYTES since pypdf holds them in memory
    monkeypatch.setattr(paper_service, "MAX_SOURCE_BYTES", 1024 * 1024)
    data = gzip.compress(b"%PDF-1.4\n" + b"\0" * (20 * 1024 * 1024))
    with pytest.raises(paper_service.SourceTooLargeError):
        paper_service._as_pdf_bytes(data)


def test_normal_tar_gz_extracts():
    data = make_tar_gz({"main.tex": TEX, "sec/intro.tex": b"Intro text", "fig.png": b"png"})
    content = paper_service._extract_tex_from_tar(data)
    assert "% === main.tex ===" in content
    assert "Hello" in content
    assert "Intro text" in content
    assert "png" not in content


def test_tex_reading_stops_past_content_cap(monkeypatch):
    monkeypatch.setattr(paper_service, "MAX_CONTENT_CHARS", 100)
    data = make_tar_gz({"a.tex": b"a" * 500, "b.tex": b"b" * 50})
    content = paper_service._extract_tex_from_tar(data)
    assert len(content) > 100
    assert "b.tex" not in content


def test_extraction_runs_in_thread(monkeypatch):
    offloaded = []
    real_to_thread = asyncio.to_thread

    async def spy_to_thread(func, *args, **kwargs):
        offloaded.append(func.__name__)
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(paper_service.asyncio, "to_thread", spy_to_thread)
    with aioresponses() as m:
        m.get(SRC_URL, status=200, body=make_tar_gz({"main.tex": TEX}))
        assert "Hello" in fetch()
    assert offloaded == ["_as_pdf_bytes", "_extract_tex_from_tar"]
