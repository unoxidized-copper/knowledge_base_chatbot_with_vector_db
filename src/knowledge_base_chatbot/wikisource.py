from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from time import sleep
from typing import Iterable
from urllib.parse import quote, unquote, urlparse

import requests
from bs4 import BeautifulSoup

from .config import BookConfig


API_URL = "https://bn.wikisource.org/w/api.php"
USER_AGENT = "knowledge-base-chatbot-course-project/1.0"


@dataclass
class Chapter:
    book_title: str
    chapter: str
    section: str
    source_url: str
    text: str


def url_to_title(url: str) -> str:
    path = urlparse(url).path
    marker = "/wiki/"
    if marker not in path:
        raise ValueError(f"Not a Wikisource page URL: {url}")
    return unquote(path.split(marker, 1)[1])


def title_to_url(title: str) -> str:
    return "https://bn.wikisource.org/wiki/" + quote(title.replace(" ", "_"), safe="/(),_")


def fetch_parse_html(title: str, session: requests.Session) -> str:
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
    }

    response = None
    for attempt in range(5):
        response = session.get(API_URL, params=params, timeout=30)
        if response.status_code != 429:
            break

        retry_after = response.headers.get("Retry-After")
        wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else 5 * (attempt + 1)
        print(f"Wikisource rate limit hit. Waiting {wait_seconds}s before retrying...")
        sleep(wait_seconds)

    assert response is not None
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload["parse"]["text"]


def _clean_lines(lines: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    skip_starts = (
        "ডাউনলোড করুন",
        "লেখাগুলো",
        "এই পাতা শেষ",
        "বিষয়শ্রেণীসমূহ",
        "লুকানো বিষয়শ্রেণী",
    )
    for line in lines:
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if line.startswith(skip_starts) or "থেকে আনীত" in line:
            break
        if line in {"| | | | |", "ভাষা যোগ করুন"}:
            continue
        if line.startswith("◄") or line.endswith("►"):
            continue
        cleaned.append(line)
    return cleaned


def _text_from_html(html: str, section: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one("div.mw-parser-output") or soup

    for tag in body.select(
        "style, script, table, sup, .mw-editsection, .noprint, .metadata, .ws-noexport"
    ):
        tag.decompose()

    lines = _clean_lines(body.get_text("\n").splitlines())

    for i, line in enumerate(lines[:30]):
        if line == section:
            lines = lines[i:]
            break

    return "\n".join(lines).strip()


def _ordered_child_titles(main_html: str, root_title: str, required_text: str) -> list[str]:
    soup = BeautifulSoup(main_html, "html.parser")
    root_prefix = root_title.replace(" ", "_") + "/"
    titles: list[str] = []

    for anchor in soup.select("a[href]"):
        href = anchor.get("href", "")
        if not href.startswith("/wiki/"):
            continue
        title = unquote(href.split("/wiki/", 1)[1]).split("#", 1)[0]
        title = title.replace("_", " ")
        if not title.startswith(root_prefix.replace("_", " ")):
            continue
        if ":" in title:
            continue
        if required_text and required_text not in title:
            continue
        if title not in titles:
            titles.append(title)

    return titles


def discover_chapter_titles(book: BookConfig, session: requests.Session | None = None) -> list[str]:
    session = session or requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    root_title = url_to_title(book.root_url).replace("_", " ")
    main_html = fetch_parse_html(root_title, session)
    titles = _ordered_child_titles(main_html, root_title, book.chapter_link_must_contain)

    if not titles:
        raise RuntimeError("No chapter subpages were found from the Wikisource table of contents.")
    return titles


def fetch_chapter(book: BookConfig, title: str, session: requests.Session) -> Chapter:
    html = fetch_parse_html(title, session)
    parts = title.split("/")
    section = parts[-1].replace("_", " ")
    chapter = " — ".join(part.replace("_", " ") for part in parts[1:])
    text = _text_from_html(html, section)

    return Chapter(
        book_title=book.title,
        chapter=chapter,
        section=section,
        source_url=title_to_url(title),
        text=text,
    )


def crawl_book(book: BookConfig, polite_delay: float = 1.0, cache_path: Path | None = None) -> list[Chapter]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    titles = discover_chapter_titles(book, session=session)
    chapters = load_chapters(cache_path) if cache_path and cache_path.exists() else []
    crawled_urls = {chapter.source_url for chapter in chapters}

    print(f"Discovered {len(titles)} chapter subpages.", flush=True)
    if chapters:
        print(f"Loaded {len(chapters)} cached chapters. Resuming crawl.", flush=True)

    for position, title in enumerate(titles, start=1):
        source_url = title_to_url(title)
        if source_url in crawled_urls:
            continue

        print(f"[{position}/{len(titles)}] Fetching {title.split('/')[-1]}", flush=True)
        chapter = fetch_chapter(book, title, session)
        if len(chapter.text) > 100:
            chapters.append(chapter)
            if cache_path:
                save_chapters(chapters, cache_path)
        sleep(polite_delay)
    return chapters


def save_chapters(chapters: list[Chapter], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump([asdict(chapter) for chapter in chapters], file, ensure_ascii=False, indent=2)


def load_chapters(path: Path) -> list[Chapter]:
    with path.open("r", encoding="utf-8") as file:
        rows = json.load(file)
    return [Chapter(**row) for row in rows]
