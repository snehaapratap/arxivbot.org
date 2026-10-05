"""Tests for arXiv fetch retries and error recovery. No real network."""

import asyncio
import re
import time

import aiohttp
import pytest
from aiohttp.http_exceptions import ContentLengthError
from aioresponses import aioresponses

from arxivbot.services import paper_service
from arxivbot.utils.arxiv import parse_arxiv_id

PAPER_ID = "2301.00001"
SRC_URL = parse_arxiv_id(PAPER_ID).src_url
TEX = b"\\documentclass{article}\\begin{document}Hello\\end{document}"


@pytest.fixture
def sleeps(monkeypatch):
    """Record backoff delays instead of sleeping."""
    calls: list[float] = []

    async def fake_sleep(delay):
        calls.append(delay)

    monkeypatch.setattr(paper_service, "_sleep", fake_sleep)
    return calls


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


def fetch():
    return asyncio.run(paper_service.fetch_paper_content(PAPER_ID))


def request_count(m):
    return sum(len(v) for v in m.requests.values())


def test_406_then_200_succeeds(sleeps):
    with aioresponses() as m:
        m.get(SRC_URL, status=406)
        m.get(SRC_URL, status=200, body=TEX)
        content = fetch()
    assert "Hello" in content
    assert sleeps == [2]
    assert paper_service._indexing_status[PAPER_ID]["status"] == "complete"


def test_content_length_error_then_200_succeeds(sleeps):
    with aioresponses() as m:
        m.get(SRC_URL, exception=ContentLengthError("Not enough data to satisfy content length header."))
        m.get(SRC_URL, status=200, body=TEX)
        content = fetch()
    assert "Hello" in content
    assert sleeps == [2]


def test_404_is_fatal_without_retry(sleeps):
    with aioresponses() as m:
        m.get(SRC_URL, status=404)
        with pytest.raises(ValueError, match=re.escape(paper_service.NOT_ON_ARXIV_MESSAGE)):
            fetch()
        assert request_count(m) == 1
    assert sleeps == []
    status = paper_service._indexing_status[PAPER_ID]
    assert status["status"] == "error"
    assert status["error"] == paper_service.NOT_ON_ARXIV_MESSAGE


def test_persistent_503_gives_up_with_friendly_error(sleeps):
    with aioresponses() as m:
        m.get(SRC_URL, status=503, repeat=True)
        with pytest.raises(ValueError) as exc:
            fetch()
        assert request_count(m) == 4  # 1 attempt + 3 retries
    assert sleeps == [2, 5, 15]
    assert str(exc.value) == paper_service.TRANSIENT_ERROR_MESSAGE
    status = paper_service._indexing_status[PAPER_ID]
    assert status["status"] == "error"
    assert status["error"] == paper_service.TRANSIENT_ERROR_MESSAGE
    assert "error_at" in status


def test_raw_exception_text_not_exposed(sleeps):
    with aioresponses() as m:
        m.get(SRC_URL, exception=aiohttp.ClientPayloadError("secret raw detail"), repeat=True)
        with pytest.raises(ValueError):
            fetch()
    assert "secret raw detail" not in paper_service._indexing_status[PAPER_ID]["error"]


def test_retry_after_honored_and_capped(sleeps):
    with aioresponses() as m:
        m.get(SRC_URL, status=429, headers={"Retry-After": "30"})
        m.get(SRC_URL, status=503, headers={"Retry-After": "600"})
        m.get(SRC_URL, status=200, body=TEX)
        fetch()
    assert sleeps == [30, paper_service.MAX_RETRY_AFTER]


def test_user_agent_sent(sleeps):
    with aioresponses() as m:
        m.get(SRC_URL, status=200, body=TEX)
        fetch()
        (call,) = next(iter(m.requests.values()))
    headers = call.kwargs.get("headers") or {}
    assert headers.get("User-Agent") == paper_service.ARXIV_USER_AGENT


# --- /api/index?auto=1 recovery -----------------------------------------------------


def test_auto_index_error_after_cooldown_retriggers_indexing(client, monkeypatch):
    calls = []

    async def fake_index(paper_id):
        calls.append(paper_id)

    monkeypatch.setattr(paper_service, "index_paper", fake_index)
    paper_service._indexing_status[PAPER_ID] = {
        "status": "error",
        "error": paper_service.TRANSIENT_ERROR_MESSAGE,
        "error_at": time.time() - paper_service.ERROR_RETRY_COOLDOWN - 1,
    }

    resp = client.post(f"/api/index/{PAPER_ID}?auto=1")
    assert resp.status_code == 200
    assert resp.json() == {"message": "Indexing started"}
    assert calls == [PAPER_ID]
    assert client.get(f"/api/status/{PAPER_ID}").json() == {
        "status": "starting",
        "progress": 5,
        "error": None,
    }


def test_auto_index_error_within_cooldown_does_not_retrigger(client, monkeypatch):
    calls = []

    async def fake_index(paper_id):
        calls.append(paper_id)

    monkeypatch.setattr(paper_service, "index_paper", fake_index)
    paper_service._indexing_status[PAPER_ID] = {
        "status": "error",
        "error": paper_service.TRANSIENT_ERROR_MESSAGE,
        "error_at": time.time(),
    }

    resp = client.post(f"/api/index/{PAPER_ID}?auto=1")
    assert resp.status_code == 200
    assert calls == []
    resp = client.get(f"/api/status/{PAPER_ID}")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "error",
        "progress": 0,
        "error": paper_service.TRANSIENT_ERROR_MESSAGE,
    }
    assert calls == []
