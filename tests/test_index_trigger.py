"""GET /api/status is read-only; POST /api/index is the only indexing trigger (v2i)."""

import time

import pytest

from arxivbot.services import paper_service

PAPER_ID = "2301.00002"
INDEX_LIMIT = 10


@pytest.fixture(autouse=True)
def clean_state():
    paper_service._indexing_status.clear()
    paper_service._content_cache.clear()
    yield
    paper_service._indexing_status.clear()
    paper_service._content_cache.clear()


@pytest.fixture
def index_calls(monkeypatch):
    """Record index_paper calls instead of hitting arXiv."""
    calls = []

    async def fake_index(paper_id):
        calls.append(paper_id)

    monkeypatch.setattr(paper_service, "index_paper", fake_index)
    return calls


def set_error(age):
    paper_service._indexing_status[PAPER_ID] = {
        "status": "error",
        "error": paper_service.TRANSIENT_ERROR_MESSAGE,
        "error_at": time.time() - age,
    }


def test_get_status_is_read_only(client, index_calls):
    for _ in range(2):
        resp = client.get(f"/api/status/{PAPER_ID}")
        assert resp.status_code == 200
        assert resp.json() == {"status": "not_started", "progress": 0, "error": None}
    assert index_calls == []
    assert PAPER_ID not in paper_service._indexing_status


def test_get_status_does_not_retry_error_after_cooldown(client, index_calls):
    set_error(paper_service.ERROR_RETRY_COOLDOWN + 1)
    resp = client.get(f"/api/status/{PAPER_ID}")
    assert resp.json()["status"] == "error"
    assert index_calls == []


def test_auto_index_starts_when_not_started(client, index_calls):
    resp = client.post(f"/api/index/{PAPER_ID}?auto=1")
    assert resp.status_code == 200
    assert resp.json() == {"message": "Indexing started"}
    assert index_calls == [PAPER_ID]
    assert client.get(f"/api/status/{PAPER_ID}").json()["status"] == "starting"


def test_auto_index_does_not_restart_in_progress(client, index_calls):
    client.post(f"/api/index/{PAPER_ID}?auto=1")
    resp = client.post(f"/api/index/{PAPER_ID}?auto=1")
    assert resp.json() == {"message": "Indexing in progress"}
    assert index_calls == [PAPER_ID]


def test_auto_index_error_within_cooldown_does_not_start(client, index_calls):
    set_error(0)
    resp = client.post(f"/api/index/{PAPER_ID}?auto=1")
    assert resp.status_code == 200
    assert index_calls == []
    assert client.get(f"/api/status/{PAPER_ID}").json()["status"] == "error"


def test_auto_index_error_after_cooldown_starts(client, index_calls):
    set_error(paper_service.ERROR_RETRY_COOLDOWN + 1)
    resp = client.post(f"/api/index/{PAPER_ID}?auto=1")
    assert resp.status_code == 200
    assert index_calls == [PAPER_ID]


def test_manual_index_starts_on_error_within_cooldown(client, index_calls):
    set_error(0)
    resp = client.post(f"/api/index/{PAPER_ID}")
    assert resp.status_code == 200
    assert resp.json() == {"message": "Indexing started"}
    assert index_calls == [PAPER_ID]


def test_index_is_rate_limited(client, index_calls):
    headers = {"Fly-Client-IP": "203.0.113.1"}
    for _ in range(INDEX_LIMIT):
        assert client.post("/api/index/bogus", headers=headers).status_code == 400
    assert client.post("/api/index/bogus", headers=headers).status_code == 429
    other = {"Fly-Client-IP": "203.0.113.2"}
    assert client.post("/api/index/bogus", headers=other).status_code == 400
    assert index_calls == []


def test_abs_page_triggers_indexing_with_auto_post(client, monkeypatch, index_calls):
    async def fetch_paper_metadata(paper_id):
        return {"title": "Attention Is All You Need", "authors": [], "abstract": ""}

    monkeypatch.setattr(paper_service, "fetch_paper_metadata", fetch_paper_metadata)
    resp = client.get("/abs/1706.03762")
    assert resp.status_code == 200
    assert "fetch(`/api/index/${paperId}?auto=1`, { method: 'POST' })" in resp.text
    assert index_calls == []


@pytest.mark.parametrize("failure", ["network", "429"])
def test_page_retries_failed_index_start_with_a_cap(failure):
    """Run the page's polling functions with a fake clock and fetch, without HTTP."""
    import shutil
    import subprocess
    from pathlib import Path

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    template = (Path(__file__).parents[1] / "templates/chat.html").read_text()
    start = template.index("function autoResize()")
    end = template.index("function requestErrorText(", start)
    state_start = template.index("let indexStartAttempts =")
    state_end = template.index("// Paper pane elements", state_start)
    script = template[state_start:state_end] + r'''
const assert = require('node:assert/strict');
const paperId = '2301.00002';
const element = () => ({style: {}, disabled: true, textContent: ''});
const statusBanner = element(), statusText = element(), progressFill = element();
const statusRetryBtn = element(), messageInput = element(), sendBtn = element();
let now = 0, timers = [], posts = [], state = 'not_started', failCount = 1;
Date.now = () => now;
const setTimeout = (fn, delay) => timers.push({fn, at: now + delay});
console.error = () => {};
async function fetch(url) {
    if (url.startsWith('/api/status/')) {
        return {json: async () => ({status: state, progress: 0, error: null})};
    }
    posts.push(url);
    if (failCount-- > 0) {
        if (process.argv[1] === 'network') throw new Error('synthetic network failure');
        return {ok: false, status: 429};
    }
    state = 'starting';
    return {ok: true, status: 200};
}
const settle = () => new Promise(resolve => setImmediate(resolve));
async function advance(target) {
    while (timers.length && timers[0].at <= target) {
        const timer = timers.shift();
        now = timer.at;
        await timer.fn();
        await settle();
    }
    now = target;
}
''' + template[start:end] + r'''
(async () => {
    await startIndexing(); await settle();
    assert.equal(posts.length, 1);
    await advance(9999);
    assert.equal(posts.length, 1, 'do not retry before ten seconds');
    await advance(10000);
    assert.equal(posts.length, 2, 'recover automatically from the failed first POST');
    assert.equal(posts[1], `/api/index/${paperId}?auto=1`);
    await advance(24000);
    assert.equal(posts.length, 2, 'do not restart indexing already in progress');
    state = 'complete';
    await advance(26000);
    assert.equal(statusBanner.style.display, 'none');
    assert.equal(messageInput.disabled, false);
    assert.equal(sendBtn.disabled, false);
    assert.equal(timers.length, 0);

    // A new page load cannot keep making automatic POSTs forever.
    state = 'not_started'; failCount = Infinity; posts = []; timers = []; now = 0;
    indexStartAttempts = 0; lastIndexStartAt = 0;
    await startIndexing(); await settle();
    await advance(30000);
    assert.equal(posts.length, 3, 'initial attempt plus two automatic retries');
    assert.equal(statusRetryBtn.style.display, 'inline-block');
    assert.equal(statusRetryBtn.disabled, false);
    assert.equal(progressFill.style.width, '0%');
    assert.match(statusText.textContent, /try again/i);
    assert.equal(timers.length, 0, 'stop polling when the retry budget is exhausted');

    // The existing manual Retry action can recover after the automatic budget.
    failCount = 0;
    await retryIndexing(); await settle();
    assert.equal(posts.at(-1), `/api/index/${paperId}`);
    assert.equal(statusRetryBtn.style.display, 'none');
    state = 'complete';
    await advance(32000);
    assert.equal(statusBanner.style.display, 'none');
})().catch(error => { console.log(error.stack); process.exitCode = 1; });
'''
    result = subprocess.run([node, "-e", script, failure], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
