"""One guide voice across status, error and empty-state copy (kfq, Personality pattern).

The voice matches plain-mode answers: short, friendly, plain words, says
what happened and what to do next. No "Error:" prefixes, no "Failed to",
no "Please", no raw HTTP phrases, and the product is always "ArxivBot".
"""

from pathlib import Path

import pytest

from arxivbot import main
from arxivbot.routes import api
from arxivbot.services import paper_service

TEMPLATES = Path(__file__).resolve().parent.parent / "templates"

COLD_PHRASES = ("Error:", "Failed to", "Please", "Rate limited", "unavailable for", "Processing...")

PAPER_MESSAGES = {
    name: getattr(paper_service, name)
    for name in (
        "TRANSIENT_ERROR_MESSAGE",
        "GENERIC_ERROR_MESSAGE",
        "SOURCE_TOO_LARGE_MESSAGE",
        "NO_TEX_SOURCE_MESSAGE",
        "SCANNED_PDF_MESSAGE",
        "UNREADABLE_PDF_MESSAGE",
        "NOT_ON_ARXIV_MESSAGE",
    )
}


def assert_guide_voice(text):
    for phrase in COLD_PHRASES:
        assert phrase not in text, f"{phrase!r} in {text!r}"
    assert "arxivbot " not in text and "arxivbot can" not in text, text
    assert text.rstrip().endswith((".", "?")), text


@pytest.mark.parametrize("name", sorted(PAPER_MESSAGES))
def test_paper_status_messages_use_guide_voice(name):
    assert_guide_voice(PAPER_MESSAGES[name])


def test_messages_say_what_to_do_next():
    # Every retryable failure tells the reader they can try again.
    for name in ("TRANSIENT_ERROR_MESSAGE", "GENERIC_ERROR_MESSAGE"):
        assert "try again" in PAPER_MESSAGES[name]
    # Dead ends point at the PDF instead.
    for name in ("SOURCE_TOO_LARGE_MESSAGE", "NO_TEX_SOURCE_MESSAGE", "SCANNED_PDF_MESSAGE",
                 "UNREADABLE_PDF_MESSAGE"):
        assert "PDF" in PAPER_MESSAGES[name], name


def test_answer_failure_message_uses_guide_voice():
    text = api._user_error_message(RuntimeError("provider exploded"))
    assert_guide_voice(text)
    assert "provider" not in text
    assert "again" in text


def test_not_found_message_uses_guide_voice():
    assert_guide_voice(main.NOT_FOUND_MESSAGE)


def test_other_error_pages_use_friendly_heading(client):
    resp = client.post("/abs/1706.03762")
    assert resp.status_code == 405
    assert "Method Not Allowed" not in resp.text
    assert "That did not work" not in resp.text
    assert "Something went wrong" in resp.text
    assert 'id="paper-form"' in resp.text


@pytest.mark.parametrize("name", ["chat.html", "error.html", "home.html", "_paper_form.html"])
def test_templates_have_no_cold_copy(name):
    text = (TEMPLATES / name).read_text()
    for phrase in COLD_PHRASES:
        assert phrase not in text, f"{phrase!r} in {name}"
    # Loading and status lines read like a guide, not a progress log.
    assert "Indexing paper..." not in text
    assert "Downloading TeX source..." not in text
    assert "Extracting TeX files..." not in text


def test_chat_api_error_detail_is_not_shown_raw():
    text = (TEMPLATES / "chat.html").read_text()
    # Validation and mismatch details are developer text; the UI maps them.
    assert "d.msg" not in text
    assert "`Error: ${" not in text
