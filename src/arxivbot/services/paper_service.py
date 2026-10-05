"""Paper service for fetching arXiv papers and answering questions via direct context."""

from __future__ import annotations

import asyncio
import gzip
import io
import logging
import re
import tarfile
import time
import zlib
from datetime import datetime
from typing import TYPE_CHECKING

import aiohttp

from arxivbot.config import get_settings
from arxivbot.services import db
from arxivbot.utils.arxiv import parse_arxiv_id

if TYPE_CHECKING:
    from litellm import Router

logger = logging.getLogger(__name__)

# Track indexing status per paper
_indexing_status: dict[str, dict] = {}

# Cache for paper content (in-memory, per paper)
_content_cache: dict[str, str] = {}

# Cache for ar5iv HTML (in-memory, per paper)
_ar5iv_cache: dict[str, str] = {}

# Maximum content size to prevent OOM (500KB should be plenty for most papers)
MAX_CONTENT_CHARS = 500_000

# Maximum downloaded source size (compressed bytes)
MAX_SOURCE_BYTES = 50 * 1024 * 1024

# Maximum total decompressed bytes read from a gzipped source (tar or single file)
MAX_DECOMPRESSED_BYTES = 200 * 1024 * 1024

# Download read chunk size
_READ_CHUNK_BYTES = 64 * 1024

# Maximum PDF pages to extract text from (PDF-only submissions)
MAX_PDF_PAGES = 300

# Maximum number of papers to cache (LRU-style, oldest evicted first)
MAX_CACHE_SIZE = 5

# Retries per model before the Router moves to the next fallback. Kept low:
# with a fallback chain, switching models beats waiting on a rate-limited one.
LLM_NUM_RETRIES = 1

# Maximum ar5iv HTML cache size
MAX_AR5IV_CACHE_SIZE = 3

# --- arXiv HTTP ---------------------------------------------------------------

ARXIV_USER_AGENT = "arxivbot/0.1 (+https://arxivbot.org)"

# Total request timeouts (seconds)
SOURCE_TIMEOUT = 60
METADATA_TIMEOUT = 20
HTML_TIMEOUT = 30

# Backoff delays between retries; len() is the max retry count
RETRY_BACKOFF = (2, 5, 15)
MAX_RETRY_AFTER = 60
RETRYABLE_STATUSES = frozenset({406, 429}) | frozenset(range(500, 600))

# Seconds after an error before /api/status re-triggers indexing
ERROR_RETRY_COOLDOWN = 60

# User-facing error messages
TRANSIENT_ERROR_MESSAGE = "arXiv isn't answering right now. Give it a minute, then try again."
GENERIC_ERROR_MESSAGE = (
    "ArxivBot couldn't get this paper from arXiv. Give it a minute, then try again."
)
SOURCE_TOO_LARGE_MESSAGE = (
    "This paper is too big for ArxivBot to read. You can still open the PDF on arXiv."
)
NOT_ON_ARXIV_MESSAGE = "ArxivBot couldn't find this paper on arXiv. Check the ID and try again."

# Cap concurrent arXiv source downloads
_download_semaphore = asyncio.Semaphore(2)

# Patchable so tests don't actually wait
_sleep = asyncio.sleep

_TRANSIENT_EXCEPTIONS = (
    asyncio.TimeoutError,
    aiohttp.ClientPayloadError,  # "Response payload is not completed"
    aiohttp.http_exceptions.ContentLengthError,
    aiohttp.ClientConnectionError,  # includes ServerDisconnectedError
)


class ArxivUnavailableError(Exception):
    """arXiv kept failing transiently after all retries."""


class PaperFetchError(ValueError):
    """Fetch failure whose message is safe to show to users."""


class PaperNotFound(Exception):
    """arXiv answered successfully and has no paper (or version) with this ID."""


class SourceTooLargeError(Exception):
    """A download or decompressed source exceeded its size cap. Never retried."""


def _retry_after_seconds(value: str | None) -> float | None:
    """Parse a numeric Retry-After header, capped. HTTP-date values are ignored."""
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    if seconds < 0:
        return None
    return min(seconds, MAX_RETRY_AFTER)


async def _arxiv_get(
    url: str,
    *,
    timeout: float,
    as_text: bool = False,
    max_retries: int = len(RETRY_BACKOFF),
    semaphore: asyncio.Semaphore | None = None,
    max_bytes: int | None = None,
) -> tuple[int, bytes | str | None]:
    """GET an arXiv URL with User-Agent, timeout and retries on transient failures.

    Returns (status, body). Body is None for non-200 responses. Non-retryable
    statuses (e.g. 404) are returned immediately. Raises ArxivUnavailableError
    once retries are exhausted. If max_bytes is set, raises SourceTooLargeError
    (without retrying) once the body is known to exceed it.
    """
    headers = {"User-Agent": ARXIV_USER_AGENT}
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    attempt = 0
    while True:
        retry_after = None
        try:
            if semaphore is not None:
                await semaphore.acquire()
            try:
                async with aiohttp.ClientSession(timeout=client_timeout) as session:
                    async with session.get(url, headers=headers, allow_redirects=True) as resp:
                        status = resp.status
                        if status == 200:
                            if max_bytes is None:
                                body = await resp.text() if as_text else await resp.read()
                            else:
                                body = await _read_capped(resp, max_bytes)
                                if as_text:
                                    body = body.decode(resp.charset or "utf-8", errors="replace")
                            return status, body
                        if status not in RETRYABLE_STATUSES:
                            return status, None
                        retry_after = _retry_after_seconds(resp.headers.get("Retry-After"))
                        reason = f"HTTP {status}"
            finally:
                if semaphore is not None:
                    semaphore.release()
        except _TRANSIENT_EXCEPTIONS as e:
            reason = f"{type(e).__name__}: {e}"

        if attempt >= max_retries:
            logger.error(f"arXiv GET {url} failed after {attempt + 1} attempts: {reason}")
            raise ArxivUnavailableError(reason)

        delay = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
        if retry_after is not None:
            delay = max(delay, retry_after)
        attempt += 1
        logger.warning(
            f"arXiv GET {url} transient failure ({reason}); retry {attempt}/{max_retries} in {delay}s"
        )
        await _sleep(delay)


async def _read_capped(resp: aiohttp.ClientResponse, max_bytes: int) -> bytes:
    """Read a response body, raising SourceTooLargeError once it exceeds max_bytes."""
    if resp.content_length is not None and resp.content_length > max_bytes:
        raise SourceTooLargeError(f"Content-Length {resp.content_length} > {max_bytes}")
    buf = bytearray()
    async for chunk in resp.content.iter_chunked(_READ_CHUNK_BYTES):
        buf += chunk
        if len(buf) > max_bytes:
            raise SourceTooLargeError(f"body exceeded {max_bytes} bytes")
    return bytes(buf)


# Router setup for primary + fallback LLM providers
def _free_tier_rpm(model: str) -> int:
    """Free-tier requests/minute for a Gemini model (Flash Lite 15, others 5)."""
    return 15 if "flash-lite" in model else 5


def _create_router() -> Router:
    """Create a litellm Router: primary, then free-tier fallbacks, then the paid key.

    Each free-tier model has its own quota, so falling back across models on
    the same key keeps chat working after one model's daily limit is used up.
    """
    # Imported lazily: litellm is slow to import and only needed to answer a chat.
    from litellm import Router

    settings = get_settings()
    model_list = [
        {
            "model_name": "gemini-primary",
            "litellm_params": {
                "model": settings.llm_model,
                "api_key": settings.gemini_api_key,
                "rpm": _free_tier_rpm(settings.llm_model),
            },
        },
    ]
    fallback_names = []
    for i, model in enumerate(settings.free_fallback_models, start=1):
        name = f"gemini-free-{i}"
        fallback_names.append(name)
        model_list.append(
            {
                "model_name": name,
                "litellm_params": {
                    "model": model,
                    "api_key": settings.gemini_api_key,
                    "rpm": _free_tier_rpm(model),
                },
            }
        )
    # Only add the paid fallback if a paid key is configured
    if settings.gemini_api_key_paid:
        fallback_names.append("gemini-fallback")
        model_list.append(
            {
                "model_name": "gemini-fallback",
                "litellm_params": {
                    "model": settings.llm_fallback_model,
                    "api_key": settings.gemini_api_key_paid,
                    "rpm": 300,
                },
            }
        )
    return Router(
        model_list=model_list,
        fallbacks=[{"gemini-primary": fallback_names}] if fallback_names else [],
        num_retries=LLM_NUM_RETRIES,
        allowed_fails=1,
        cooldown_time=60,
    )


# Module-level router instance (initialized on first use)
_router: Router | None = None


def _get_router() -> Router:
    """Get or create the singleton Router instance."""
    global _router
    if _router is None:
        _router = _create_router()
    return _router


class _BoundedReader(io.RawIOBase):
    """Read-only stream wrapper that raises SourceTooLargeError past `limit` bytes."""

    def __init__(self, raw, limit: int):
        self._raw = raw
        self._limit = limit
        self._count = 0

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._limit - self._count + 1
        chunk = self._raw.read(size)
        self._count += len(chunk)
        if self._count > self._limit:
            raise SourceTooLargeError(f"decompressed source exceeded {self._limit} bytes")
        return chunk

    def readinto(self, b) -> int:
        chunk = self.read(len(b))
        b[: len(chunk)] = chunk
        return len(chunk)


def _gunzip_stream(data: bytes, limit: int | None = None) -> _BoundedReader:
    """Stream-decompress gzip data, capped at limit (default MAX_DECOMPRESSED_BYTES)."""
    return _BoundedReader(
        gzip.GzipFile(fileobj=io.BytesIO(data)), limit or MAX_DECOMPRESSED_BYTES
    )


def _char_budget_bytes(chars_left: int) -> int:
    """Bytes to read so decoding yields more than chars_left chars (UTF-8 is <= 4 bytes/char)."""
    return (max(chars_left, 0) + 1) * 4


def _extract_tex_from_tar(data: bytes) -> str:
    """Extract and concatenate .tex files from a tar.gz archive (or single .tex / .tex.gz).

    Memory-bounded: decompression is streamed and capped at MAX_DECOMPRESSED_BYTES
    (raising SourceTooLargeError), and reading stops once the collected text exceeds
    MAX_CONTENT_CHARS (the caller truncates to that anyway).
    """
    if data[:2] != GZIP_MAGIC:
        # Maybe it's a plain .tex file
        if b"\\documentclass" in data or b"\\begin{document}" in data:
            return data[: _char_budget_bytes(MAX_CONTENT_CHARS)].decode("utf-8", errors="replace")
        return ""

    tex_contents: list[str] = []
    total = 0
    try:
        # Try as tar.gz first; stream mode reads sequentially without seeking
        with tarfile.open(fileobj=_gunzip_stream(data), mode="r|") as tar:
            for member in tar:
                if member.name.endswith(".tex") and member.isfile():
                    f = tar.extractfile(member)
                    if f:
                        raw = f.read(_char_budget_bytes(MAX_CONTENT_CHARS - total))
                        content = raw.decode("utf-8", errors="replace")
                        entry = f"% === {member.name} ===\n{content}"
                        tex_contents.append(entry)
                        total += len(entry) + 2
                        if total > MAX_CONTENT_CHARS:
                            break
        return "\n\n".join(tex_contents)
    except (tarfile.ReadError, OSError, EOFError, zlib.error):
        pass

    try:
        # Maybe it's just gzipped (single file): keep a prefix, but drain the
        # rest through the bounded reader so gzip bombs are rejected
        stream = _gunzip_stream(data)
        raw = stream.read(_char_budget_bytes(MAX_CONTENT_CHARS))
        while stream.read(_READ_CHUNK_BYTES):
            pass
        return raw.decode("utf-8", errors="replace")
    except (OSError, EOFError, zlib.error):
        return ""


PDF_MAGIC = b"%PDF-"
GZIP_MAGIC = b"\x1f\x8b"

NO_TEX_SOURCE_MESSAGE = (
    "arXiv doesn't have a readable version of this paper. You can still open the PDF on arXiv."
)
SCANNED_PDF_MESSAGE = (
    "This paper is a scanned PDF, which ArxivBot can't read yet. You can still open it on arXiv."
)
UNREADABLE_PDF_MESSAGE = (
    "This paper only comes as a PDF, and ArxivBot couldn't read it. You can still open it on arXiv."
)


def _as_pdf_bytes(data: bytes) -> bytes | None:
    """Return PDF bytes if data is a PDF (raw or gzipped), else None.

    Raises SourceTooLargeError if a gzipped PDF exceeds MAX_SOURCE_BYTES once
    decompressed (the whole PDF is held in memory for pypdf).
    """
    if data[:5] == PDF_MAGIC:
        return data
    if data[:2] == GZIP_MAGIC:
        try:
            # Peek first so tar.gz sources aren't fully decompressed here
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
                if gz.read(5) != PDF_MAGIC:
                    return None
            stream = _gunzip_stream(data, MAX_SOURCE_BYTES)
            parts = []
            while chunk := stream.read(_READ_CHUNK_BYTES):
                parts.append(chunk)
            return b"".join(parts)
        except (OSError, EOFError, zlib.error):
            return None
    return None


def _extract_text_from_pdf(data: bytes) -> str:
    """Extract text from a PDF, stopping past MAX_CONTENT_CHARS or MAX_PDF_PAGES.

    Raises PaperFetchError if the PDF is unreadable or has no extractable text.
    """
    parts: list[str] = []
    total = 0
    try:
        import pypdf  # lazy: only needed for the PDF-text fallback

        reader = pypdf.PdfReader(io.BytesIO(data))
        for i, page in enumerate(reader.pages):
            if i >= MAX_PDF_PAGES:
                break
            text = page.extract_text() or ""
            if text.strip():
                parts.append(text)
                total += len(text)
            if total > MAX_CONTENT_CHARS:
                break
    except Exception as e:
        logger.warning(f"PDF text extraction failed: {e!r}")
        raise PaperFetchError(UNREADABLE_PDF_MESSAGE) from e

    if not parts:
        raise PaperFetchError(SCANNED_PDF_MESSAGE)
    return "\n\n".join(parts)


async def get_indexing_status(paper_id: str) -> dict:
    """Get the current indexing status for a paper."""
    if paper_id in _indexing_status:
        return _indexing_status[paper_id]

    # Check if already in cache
    if paper_id in _content_cache:
        return {"status": "complete", "progress": 100}

    # Check if paper exists in database
    paper = await db.get_paper(paper_id)
    if paper and paper.indexed_at:
        return {"status": "complete", "progress": 100}

    return {"status": "not_started", "progress": 0}


def should_retry_after_error(status: dict) -> bool:
    """True if an error status is old enough that indexing may be re-attempted."""
    if status.get("status") != "error":
        return False
    error_at = status.get("error_at")
    if error_at is None:
        return True
    return time.time() - error_at >= ERROR_RETRY_COOLDOWN


def mark_indexing_started(paper_id: str) -> dict:
    """Record that indexing was scheduled, so concurrent polls don't re-trigger it."""
    status = {"status": "starting", "progress": 5}
    _indexing_status[paper_id] = status
    return status


async def fetch_paper_content(paper_id: str) -> str:
    """Fetch a paper's source. Returns concatenated .tex content, or PDF text if PDF-only."""
    parsed = parse_arxiv_id(paper_id)
    if not parsed:
        raise ValueError(f"Invalid arXiv ID: {paper_id}")
    paper_id = parsed.paper_id

    # Return cached content if available
    if paper_id in _content_cache:
        logger.info(f"Paper {paper_id} content cached")
        return _content_cache[paper_id]

    _indexing_status[paper_id] = {"status": "fetching", "progress": 10}

    try:
        _indexing_status[paper_id] = {"status": "downloading", "progress": 30}

        try:
            status, data = await _arxiv_get(
                parsed.src_url,
                timeout=SOURCE_TIMEOUT,
                semaphore=_download_semaphore,
                max_bytes=MAX_SOURCE_BYTES,
            )
        except ArxivUnavailableError as e:
            raise PaperFetchError(TRANSIENT_ERROR_MESSAGE) from e
        except SourceTooLargeError as e:
            raise PaperFetchError(SOURCE_TOO_LARGE_MESSAGE) from e
        if status == 404:
            raise PaperFetchError(NOT_ON_ARXIV_MESSAGE)
        if status != 200:
            logger.error(f"arXiv source for {paper_id} returned HTTP {status}")
            raise PaperFetchError(GENERIC_ERROR_MESSAGE)

        _indexing_status[paper_id] = {"status": "extracting", "progress": 60}

        # Decompression/extraction is CPU-bound; keep it off the event loop
        try:
            pdf_data = await asyncio.to_thread(_as_pdf_bytes, data)
            if pdf_data is not None:
                logger.info(f"Paper {paper_id} source is PDF-only; extracting text")
                data = None  # drop the original buffer before extraction
                content = await asyncio.to_thread(_extract_text_from_pdf, pdf_data)
            else:
                content = await asyncio.to_thread(_extract_tex_from_tar, data)
                data = None
                if not content:
                    raise PaperFetchError(NO_TEX_SOURCE_MESSAGE)
        except SourceTooLargeError as e:
            raise PaperFetchError(SOURCE_TOO_LARGE_MESSAGE) from e

        # Truncate very large papers to prevent OOM
        if len(content) > MAX_CONTENT_CHARS:
            logger.warning(
                f"Paper {paper_id} content truncated from {len(content):,} to {MAX_CONTENT_CHARS:,} chars"
            )
            content = content[:MAX_CONTENT_CHARS] + "\n\n[... content truncated due to size ...]"

        _indexing_status[paper_id] = {"status": "complete", "progress": 100}

        # Cache the content (with size limit to prevent OOM)
        if len(_content_cache) >= MAX_CACHE_SIZE:
            # Evict oldest entry
            oldest_key = next(iter(_content_cache))
            del _content_cache[oldest_key]
            logger.info(f"Evicted paper {oldest_key} from cache")
        _content_cache[paper_id] = content

        # Save paper metadata to database
        paper = db.Paper(
            id=paper_id,
            title=None,
            authors=None,
            abstract=None,
            indexed_at=datetime.now(),
        )
        await db.upsert_paper(paper)

        logger.info(f"Successfully fetched paper {paper_id} ({len(content):,} chars)")
        return content

    except PaperFetchError as e:
        _indexing_status[paper_id] = {"status": "error", "error": str(e), "error_at": time.time()}
        logger.error(f"Failed to fetch paper {paper_id}: {e!r} (cause: {e.__cause__!r})")
        raise
    except Exception as e:
        _indexing_status[paper_id] = {
            "status": "error",
            "error": GENERIC_ERROR_MESSAGE,
            "error_at": time.time(),
        }
        logger.exception(f"Failed to fetch paper {paper_id}: {e!r}")
        raise PaperFetchError(GENERIC_ERROR_MESSAGE) from e


async def index_paper(paper_id: str) -> str:
    """Alias for fetch_paper_content for API compatibility."""
    return await fetch_paper_content(paper_id)


LEVEL_INSTRUCTIONS = {
    "plain": """Audience: a curious reader with no research background.
- Open with a one or two sentence answer in everyday words, then add detail
- Explain every technical term the first time you use it, in plain words
- Use a concrete analogy or example when an idea is abstract
- Avoid equations unless asked; describe what they mean instead""",
    "technical": """Audience: a reader comfortable with the field's terminology.
- Be precise and use the paper's own terms, notation and equations where useful
- Point to specific sections, experiments and numbers""",
}

FOLLOW_UP_HEADING = "**Ask next:**"

# The system prompt from before the lay-reader pass (nf4), verbatim. Used by
# the /old snapshot UI (level="legacy") so before/after demos are like for like.
LEGACY_SYSTEM_PROMPT = """You are a helpful research assistant. Answer questions about the following scientific paper based on its source (LaTeX, or text extracted from its PDF).

<paper>
{content}
</paper>

Instructions:
- Answer questions accurately based on the paper content
- When referencing specific text from the paper, quote it using this exact format on its own line:
  > "exact text from the paper"
- Keep quotes concise (under 100 characters when possible)
- Quote the exact wording from the paper, not a paraphrase
- If something isn't in the paper, say so
- Be concise but thorough

Example response format:
The authors propose a novel approach to optimization. As stated in the paper:
> "our method achieves 95% accuracy on the benchmark"
This represents a significant improvement over prior work."""




def _build_messages(
    content: str,
    question: str,
    chat_history: list[dict] | None = None,
    level: str = "plain",
) -> list[dict]:
    """Build the LLM message list for querying a paper."""
    if level == "legacy":
        system = LEGACY_SYSTEM_PROMPT.replace("{content}", content)
        return _with_history(system, question, chat_history)
    audience = LEVEL_INSTRUCTIONS.get(level, LEVEL_INSTRUCTIONS["plain"])
    system = f"""You are a helpful research assistant. Answer questions about the following scientific paper based on its source (LaTeX, or text extracted from its PDF).

<paper>
{content}
</paper>

{audience}

Instructions:
- Answer questions accurately based on the paper content
- When referencing specific text from the paper, quote it using this exact format on its own line:
  > "exact text from the paper"
- Keep quotes concise (under 100 characters when possible)
- Quote the exact wording from the paper, not a paraphrase
- If something isn't in the paper, say so
- Be concise but thorough
- End every answer with the line {FOLLOW_UP_HEADING} followed by exactly three short follow-up questions the reader might ask next, each on its own "- " bullet line. Nothing may come after them.

Example response format:
The authors propose a novel approach to optimization. As stated in the paper:
> "our method achieves 95% accuracy on the benchmark"
This represents a significant improvement over prior work.

{FOLLOW_UP_HEADING}
- How does the benchmark measure accuracy?
- What did earlier methods score?
- Where does the method still fail?"""
    return _with_history(system, question, chat_history)


def _with_history(
    system: str, question: str, chat_history: list[dict] | None
) -> list[dict]:
    """System prompt, then recent history, then the new question."""
    messages = [{"role": "system", "content": system}]
    if chat_history:
        for msg in chat_history[-10:]:  # Last 10 messages for context
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": question})
    return messages


async def query_paper(
    paper_id: str,
    question: str,
    chat_history: list[dict] | None = None,
    level: str = "plain",
) -> dict:
    """
    Query a paper using direct context.

    Returns:
        dict with 'answer' key
    """
    content = await fetch_paper_content(paper_id)
    settings = get_settings()
    messages = _build_messages(content, question, chat_history, level)

    # Call LLM via Router (falls back across models on errors and rate limits)
    router = _get_router()
    response = await router.acompletion(
        model="gemini-primary",
        messages=messages,
        num_retries=LLM_NUM_RETRIES,
    )

    return {
        "answer": response.choices[0].message.content,
        "citations": [],  # No citations in direct context mode
    }


async def query_paper_stream(
    paper_id: str,
    question: str,
    chat_history: list[dict] | None = None,
    level: str = "plain",
):
    """
    Query a paper using direct context with streaming response.

    Yields:
        str chunks of the response
    """
    content = await fetch_paper_content(paper_id)
    settings = get_settings()
    messages = _build_messages(content, question, chat_history, level)

    # Call LLM via Router with streaming (falls back across models on errors and rate limits)
    router = _get_router()
    response = await router.acompletion(
        model="gemini-primary",
        messages=messages,
        stream=True,
        num_retries=LLM_NUM_RETRIES,
    )

    async for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


async def fetch_paper_metadata(paper_id: str) -> dict | None:
    """Fetch paper metadata from arXiv API.

    Raises PaperNotFound when arXiv answers 200 with no entry. Returns None for
    an invalid ID or a transient failure (non-200, timeout, unparseable reply).
    """
    import xml.etree.ElementTree as ET

    parsed = parse_arxiv_id(paper_id)
    if not parsed:
        return None

    try:
        status, text = await _arxiv_get(
            parsed.api_url, timeout=METADATA_TIMEOUT, as_text=True, max_retries=1
        )
        if status != 200:
            return None

        root = ET.fromstring(text)

        # Define namespace
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        entry = root.find("atom:entry", ns)
        if entry is None:
            raise PaperNotFound(paper_id)

        title = entry.find("atom:title", ns)
        summary = entry.find("atom:summary", ns)
        authors = entry.findall("atom:author/atom:name", ns)

        return {
            "title": title.text.strip().replace("\n", " ") if title is not None else None,
            "abstract": summary.text.strip() if summary is not None else None,
            "authors": [a.text for a in authors] if authors else [],
        }

    except PaperNotFound:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch metadata for {paper_id}: {e}")
        return None


async def fetch_ar5iv_html(paper_id: str) -> str | None:
    """
    Fetch arXiv HTML for a paper, rewrite URLs to absolute, and inject bridge script.

    Returns None if HTML is unavailable for this paper.
    """
    parsed = parse_arxiv_id(paper_id)
    if not parsed:
        return None
    paper_id = parsed.paper_id

    # Return cached HTML if available
    if paper_id in _ar5iv_cache:
        logger.info(f"arXiv HTML for {paper_id} served from cache")
        return _ar5iv_cache[paper_id]

    # Use official arXiv HTML (better rendering than ar5iv)
    arxiv_html_url = f"https://arxiv.org/html/{paper_id}"

    try:
        status, html = await _arxiv_get(
            arxiv_html_url, timeout=HTML_TIMEOUT, as_text=True, max_retries=1
        )
        if status == 404:
            logger.warning(f"arXiv HTML not available for {paper_id}")
            return None
        if status != 200:
            logger.warning(f"arXiv returned HTTP {status} for {paper_id}")
            return None

        # Rewrite relative URLs to absolute arXiv URLs
        base_url = "https://arxiv.org"

        # Remove <base> tag as it interferes with our proxy
        html = re.sub(r'<base[^>]*/?>', '', html, flags=re.IGNORECASE)

        # Rewrite href="/..." and src="/..."
        html = re.sub(
            r'(href|src)="(/[^"]*)"',
            rf'\1="{base_url}\2"',
            html,
        )

        # Rewrite url(/...) in inline styles
        html = re.sub(
            r'url\((/[^)]*)\)',
            rf'url({base_url}\1)',
            html,
        )

        # Inject our bridge script before </body>
        bridge_script = """
<script src="/static/ar5iv-bridge.js"></script>
"""
        html = html.replace('</body>', bridge_script + '</body>')

        # Inject highlight CSS in <head>
        highlight_css = """
<style>
.llm-highlight,
.llm-highlight * {
    background: #fff59d !important;
    color: #000 !important;
    outline: 2px solid #ffc107;
    scroll-margin-top: 80px;
}
.llm-highlight-pulse,
.llm-highlight-pulse * {
    animation: llm-pulse 0.5s ease-in-out 2;
}
@keyframes llm-pulse {
    0%, 100% { background: #fff59d !important; color: #000 !important; }
    50% { background: #ffeb3b !important; color: #000 !important; }
}
</style>
"""
        html = html.replace('</head>', highlight_css + '</head>')

        # Cache the HTML
        if len(_ar5iv_cache) >= MAX_AR5IV_CACHE_SIZE:
            oldest_key = next(iter(_ar5iv_cache))
            del _ar5iv_cache[oldest_key]
            logger.info(f"Evicted arXiv HTML for {oldest_key} from cache")
        _ar5iv_cache[paper_id] = html

        logger.info(f"Successfully fetched arXiv HTML for {paper_id} ({len(html):,} bytes)")
        return html

    except Exception as e:
        logger.error(f"Failed to fetch arXiv HTML for {paper_id}: {e}")
        return None
