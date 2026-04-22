#!/usr/bin/env python3
"""Download Wikipedia articles into the corpus directory.

Usage:
    python scripts/download_corpus.py                  # download default articles
    python scripts/download_corpus.py "Alan Turing" "CRISPR"   # specific articles
    python scripts/download_corpus.py --list           # show default list
    python scripts/download_corpus.py --out corpus/    # custom output directory
"""
import argparse
import re
import sys
import time
import urllib.request
import urllib.parse
import json
from pathlib import Path

# Long, factually rich Wikipedia articles on clearly distinct topics.
# Chosen to be unambiguous when confused (you can't mistake Rome for quantum mechanics).
DEFAULT_ARTICLES = [
    "History of the Internet",
    "World War II",
    "Climate change",
    "Ancient Rome",
    "Human genome",
    "Solar System",
    "Machine learning",
    "Black Death",
    "History of aviation",
    "Quantum mechanics",
]

CORPUS_DIR = Path(__file__).parent.parent / "corpus"
API_URL = "https://en.wikipedia.org/w/api.php"


def fetch_article(title: str) -> tuple[str, str]:
    """Return (title, plain-text body) for a Wikipedia article."""
    params = urllib.parse.urlencode({
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "1",
        "exsectionformat": "plain",
        "format": "json",
        "redirects": "1",
    })
    url = f"{API_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "kvcache-sanity-check/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    if "missing" in page:
        raise ValueError(f"Article not found: {title!r}")

    resolved_title = page["title"]
    text = page.get("extract", "").strip()
    if not text:
        raise ValueError(f"Empty extract for: {title!r}")
    return resolved_title, text


def slugify(title: str) -> str:
    """Convert article title to a safe filename."""
    slug = title.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return f"doc_{slug}"


def save_article(title: str, body: str, out_dir: Path) -> Path:
    filename = slugify(title) + ".txt"
    path = out_dir / filename
    path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("articles", nargs="*",
                        help="Wikipedia article titles to download (default: built-in list)")
    parser.add_argument("--list", action="store_true",
                        help="Print the default article list and exit")
    parser.add_argument("--out", default=str(CORPUS_DIR), metavar="DIR",
                        help=f"Output directory (default: {CORPUS_DIR})")
    args = parser.parse_args()

    if args.list:
        print("Default articles:")
        for a in DEFAULT_ARTICLES:
            print(f"  {a}")
        return

    articles = args.articles or DEFAULT_ARTICLES
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    for title in articles:
        try:
            print(f"Fetching: {title} ...", end=" ", flush=True)
            resolved_title, body = fetch_article(title)
            path = save_article(resolved_title, body, out_dir)
            words = len(body.split())
            print(f"OK  ({words:,} words → {path.name})")
            ok += 1
        except Exception as exc:
            print(f"FAILED: {exc}")
        time.sleep(0.5)   # be polite to Wikipedia

    print(f"\n{ok}/{len(articles)} articles saved to {out_dir}/")
    if ok < len(articles):
        sys.exit(1)


if __name__ == "__main__":
    main()
