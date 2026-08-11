"""Hacker News "Research Brief" generator.

An LLM-backed pipeline that turns the Hacker News front page into a grounded,
evidence-based research brief. The flow follows a strict layered architecture:

    Discovery  -> Candidate Filtering -> Article Extraction -> Record Validation
    -> LLM Analysis -> Application Validation -> Terminal report + Markdown artifact

Each layer has a narrow responsibility and a well-defined data contract so the
stages can be tested and reasoned about in isolation.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import sys
from urllib.parse import urljoin, urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

# --------------------------------------------------------------------------- #
# Configuration constants
# --------------------------------------------------------------------------- #
HN_BASE_URL = "https://news.ycombinator.com/"
HN_HOST = "news.ycombinator.com"
DEFAULT_MODEL = "gpt-4o-mini"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 ResearchBrief/1.0"
)
REQUEST_TIMEOUT = 15
MAX_ARTICLES = 3
MIN_CONTENT_CHARS = 250
MAX_CHARS_PER_ARTICLE = 6000
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
SEEN_PATH = os.path.join(OUTPUT_DIR, "seen_articles.json")


# --------------------------------------------------------------------------- #
# Data models (contracts between layers)
# --------------------------------------------------------------------------- #
class HackerNewsArticleMetadata(BaseModel):
    """Metadata scraped from a single Hacker News front-page row."""

    source_id: str
    rank: int
    title: str
    url: str  # absolute link to the article (external) or HN item (internal)
    author: str | None = None
    points: int | None = None
    comments: int | None = None
    hn_url: str  # absolute link to the HN discussion


class ArticleContent(BaseModel):
    """Main textual content extracted from an external article page."""

    source_id: str
    title: str
    url: str
    text: str


class ArticleInsight(BaseModel):
    """Structured, evidence-backed insights for a single article."""

    source_id: str
    title: str
    url: str = ""
    summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    sentiment: str = "neutral"
    evidence: list[str] = Field(default_factory=list)


class ResearchBrief(BaseModel):
    """The complete LLM-generated brief across the analysed article set."""

    insights: list[ArticleInsight] = Field(default_factory=list)
    overall_summary: str = ""
    cross_article_synthesis: str = ""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _parse_int(text: str) -> int | None:
    """Extract the first integer found in ``text`` (handling thousands commas)."""
    if not text:
        return None
    match = re.search(r"\d[\d,]*", text)
    return int(match.group().replace(",", "")) if match else None


def _strip_code_fence(text: str) -> str:
    """Remove a leading/trailing Markdown code fence from an LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Discovery layer: HN front page -> List[HackerNewsArticleMetadata]
# --------------------------------------------------------------------------- #
def fetch_frontpage(url: str = HN_BASE_URL) -> str:
    """Fetch the raw HTML of the Hacker News front page."""
    response = requests.get(
        url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}
    )
    response.raise_for_status()
    return response.text


def parse_frontpage(html: str) -> list[HackerNewsArticleMetadata]:
    """Parse front-page HTML into structured article metadata records."""
    soup = BeautifulSoup(html, "html.parser")
    articles: list[HackerNewsArticleMetadata] = []

    for row in soup.select("tr.athing"):
        source_id = (row.get("id") or "").strip()

        rank_el = row.select_one(".rank")
        rank = _parse_int(rank_el.get_text()) if rank_el else None

        title_link = row.select_one("span.titleline > a") or row.select_one(
            "td.title a"
        )
        if not title_link:
            continue

        title = title_link.get_text(strip=True)
        href = (title_link.get("href") or "").strip()
        url = urljoin(HN_BASE_URL, href)
        hn_url = (
            urljoin(HN_BASE_URL, f"item?id={source_id}") if source_id else HN_BASE_URL
        )

        author: str | None = None
        points: int | None = None
        comments: int | None = None

        subtext_row = row.find_next_sibling("tr")
        subtext = subtext_row.select_one("td.subtext") if subtext_row else None
        if subtext:
            score_el = subtext.select_one("span.score")
            if score_el:
                points = _parse_int(score_el.get_text())

            user_el = subtext.select_one("a.hnuser")
            if user_el:
                author = user_el.get_text(strip=True)

            for anchor in subtext.select("a"):
                label = anchor.get_text(strip=True).lower()
                if "comment" in label:
                    comments = _parse_int(label) or 0
                elif label == "discuss":
                    comments = 0

        articles.append(
            HackerNewsArticleMetadata(
                source_id=source_id,
                rank=rank if rank is not None else 0,
                title=title,
                url=url,
                author=author,
                points=points,
                comments=comments,
                hn_url=hn_url,
            )
        )

    return articles


# --------------------------------------------------------------------------- #
# Candidate filtering: keep external HTTP/HTTPS links only
# --------------------------------------------------------------------------- #
def filter_external(
    articles: list[HackerNewsArticleMetadata],
) -> list[HackerNewsArticleMetadata]:
    """Return only articles that point to an external http/https destination."""
    external: list[HackerNewsArticleMetadata] = []
    for article in articles:
        parsed = urlparse(article.url)
        if parsed.scheme in ("http", "https") and parsed.netloc.lower() != HN_HOST:
            external.append(article)
    return external


# --------------------------------------------------------------------------- #
# Deduplication: skip candidates already processed on previous runs
# --------------------------------------------------------------------------- #
def load_seen_ids(path: str = SEEN_PATH) -> set[str]:
    """Load the set of Hacker News source IDs processed on previous runs.

    Returns an empty set when the file is missing or unreadable, so a fresh
    checkout simply behaves as if nothing has been seen yet.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return set()
    raw = data.get("seen", []) if isinstance(data, dict) else data
    return {str(item) for item in raw}


def save_seen_ids(seen: set[str], path: str = SEEN_PATH) -> None:
    """Persist the set of processed source IDs to a small local JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"seen": sorted(seen)}, handle, indent=2)


def filter_unseen(
    candidates: list[HackerNewsArticleMetadata],
    seen_ids: set[str],
) -> list[HackerNewsArticleMetadata]:
    """Drop candidates whose source_id was already processed on a prior run."""
    return [
        candidate for candidate in candidates if candidate.source_id not in seen_ids
    ]


# --------------------------------------------------------------------------- #
# Extraction layer: External pages -> List[ArticleContent]
# --------------------------------------------------------------------------- #
def fetch_article_content(
    meta: HackerNewsArticleMetadata,
) -> ArticleContent | None:
    """Fetch and extract the main content of a single article, or ``None``.

    Failures (network errors, non-HTML content types, thin pages) are skipped
    rather than raised so a single bad link never aborts the pipeline.
    """
    try:
        response = requests.get(
            meta.url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}
        )
    except requests.RequestException as exc:
        print(f"    ! Skipping {meta.url}: request failed ({exc})")
        return None

    if response.status_code != 200:
        print(f"    ! Skipping {meta.url}: HTTP {response.status_code}")
        return None

    content_type = response.headers.get("Content-Type", "").lower()
    if "html" not in content_type:
        print(f"    ! Skipping {meta.url}: unsupported content type '{content_type}'")
        return None

    extracted = trafilatura.extract(
        response.text, include_comments=False, include_tables=False
    )
    text = (extracted or "").strip()
    if len(text) < MIN_CONTENT_CHARS:
        print(f"    ! Skipping {meta.url}: no usable article text extracted")
        return None

    return ArticleContent(
        source_id=meta.source_id,
        title=meta.title,
        url=meta.url,
        text=text,
    )


def extract_articles(
    candidates: list[HackerNewsArticleMetadata],
    limit: int = MAX_ARTICLES,
) -> list[ArticleContent]:
    """Fetch content for candidates, stopping after ``limit`` usable articles."""
    collected: list[ArticleContent] = []
    for meta in candidates:
        if len(collected) >= limit:
            break
        print(f"    - Fetching: {meta.title}")
        content = fetch_article_content(meta)
        if content is not None:
            collected.append(content)
    return collected


# --------------------------------------------------------------------------- #
# Record validation layer: verify extracted article records
# --------------------------------------------------------------------------- #
def validate_articles(
    articles: list[ArticleContent],
    known_source_ids: set[str],
) -> tuple[list[ArticleContent], list[str]]:
    """Validate extracted records against the set of known HN source IDs.

    Any article whose ``source_id`` is empty or does not appear in
    ``known_source_ids`` is dropped and reported as a warning, guaranteeing the
    downstream LLM only ever sees traceable, non-hallucinated source material.
    """
    valid: list[ArticleContent] = []
    warnings: list[str] = []

    for article in articles:
        if not article.source_id or article.source_id not in known_source_ids:
            warnings.append(
                f"Article '{article.title}' has unknown source_id "
                f"'{article.source_id}'; dropping record."
            )
            continue
        if not article.text.strip():
            warnings.append(
                f"Article '{article.title}' (source_id '{article.source_id}') "
                "has empty content; dropping record."
            )
            continue
        valid.append(article)

    return valid, warnings


# --------------------------------------------------------------------------- #
# LLM analysis layer: List[ArticleContent] -> ResearchBrief
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT = (
    "You are a meticulous research analyst. You read news/tech articles and "
    "produce a structured JSON research brief. Strict rules:\n"
    "1. Use ONLY information present in the provided article texts. Never invent "
    "facts, numbers, names, or events.\n"
    "2. Reuse the EXACT source_id and title supplied for each article. Do not "
    "alter, translate, or paraphrase them.\n"
    "3. Ground every key point and the sentiment in the text, and supply short "
    "verbatim evidence quotes copied from the article.\n"
    "4. If the text does not support a claim, omit the claim.\n"
    "Respond with valid JSON only - no Markdown, no commentary."
)

_SCHEMA_EXAMPLE = {
    "insights": [
        {
            "source_id": "<exact source_id>",
            "title": "<exact title>",
            "url": "<article url>",
            "summary": "<2-3 sentence neutral summary grounded in the text>",
            "key_points": ["<key point>", "<key point>"],
            "sentiment": "positive | neutral | negative",
            "evidence": ["<verbatim quote from the article>"],
        }
    ],
    "overall_summary": "<high-level summary across all articles>",
    "cross_article_synthesis": "<themes/contrasts connecting the articles>",
}


def build_client() -> OpenAI:
    """Construct an OpenAI client from the ``OPENAI_API_KEY`` environment var."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file before running."
        )
    return OpenAI(api_key=api_key)


def _call_llm(client: OpenAI, model: str, system: str, user: str) -> str:
    """Call the chat completions API, degrading gracefully across models."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
        )
    except Exception:
        # Some models reject json_object mode or custom temperatures; retry plain.
        response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content or ""


def analyze_articles(
    articles: list[ArticleContent],
    client: OpenAI,
    model: str,
) -> ResearchBrief:
    """Produce a structured :class:`ResearchBrief` from the article set."""
    payload = [
        {
            "source_id": article.source_id,
            "title": article.title,
            "url": article.url,
            "text": article.text[:MAX_CHARS_PER_ARTICLE],
        }
        for article in articles
    ]

    user_prompt = (
        "Analyse the following articles and return a JSON object with EXACTLY "
        "this shape:\n"
        f"{json.dumps(_SCHEMA_EXAMPLE, indent=2)}\n\n"
        "Produce one entry in `insights` per article, preserving each provided "
        "source_id and title verbatim.\n\n"
        f"Articles:\n{json.dumps(payload, ensure_ascii=False)}"
    )

    raw = _call_llm(client, model, _SYSTEM_PROMPT, user_prompt)
    data = json.loads(_strip_code_fence(raw))
    return ResearchBrief.model_validate(data)


# --------------------------------------------------------------------------- #
# Application validation: ResearchBrief + article set -> verified brief
# --------------------------------------------------------------------------- #
def verify_brief(
    brief: ResearchBrief,
    articles: list[ArticleContent],
) -> tuple[ResearchBrief, list[str]]:
    """Cross-check the LLM output against the ground-truth article set.

    Enforces the anti-hallucination contract: insights must reference a known
    source_id, titles are corrected to match the source exactly, missing URLs
    are backfilled, and any uncovered article is reported.
    """
    warnings: list[str] = []
    known = {article.source_id: article for article in articles}
    verified: list[ArticleInsight] = []

    for insight in brief.insights:
        source = known.get(insight.source_id)
        if source is None:
            warnings.append(
                f"Dropping insight with unknown source_id '{insight.source_id}' "
                "(not part of the source set - possible hallucination)."
            )
            continue
        if insight.title != source.title:
            warnings.append(
                f"Corrected title for source_id '{insight.source_id}' to match "
                "the source article."
            )
            insight.title = source.title
        if not insight.url:
            insight.url = source.url
        verified.append(insight)

    brief.insights = verified

    covered = {insight.source_id for insight in verified}
    for article in articles:
        if article.source_id not in covered:
            warnings.append(
                f"No insight was produced for source_id '{article.source_id}' "
                f"({article.title})."
            )

    return brief, warnings


# --------------------------------------------------------------------------- #
# Presentation layer: verified brief -> terminal report + Markdown artifact
# --------------------------------------------------------------------------- #
def render_terminal_report(brief: ResearchBrief, warnings: list[str]) -> None:
    """Print a human-readable research brief to the terminal."""
    line = "=" * 72
    print(f"\n{line}")
    print("HACKER NEWS RESEARCH BRIEF")
    print(line)

    for index, insight in enumerate(brief.insights, start=1):
        print(f"\n[{index}] {insight.title}")
        print(f"    Source ID : {insight.source_id}")
        print(f"    URL       : {insight.url}")
        print(f"    Sentiment : {insight.sentiment}")
        if insight.summary:
            print(f"    Summary   : {insight.summary}")
        if insight.key_points:
            print("    Key points:")
            for point in insight.key_points:
                print(f"      - {point}")
        if insight.evidence:
            print("    Evidence:")
            for quote in insight.evidence:
                print(f"      > {quote}")

    if brief.overall_summary:
        print(f"\n{line}")
        print("OVERALL SUMMARY")
        print(line)
        print(brief.overall_summary)

    if brief.cross_article_synthesis:
        print(f"\n{line}")
        print("CROSS-ARTICLE SYNTHESIS")
        print(line)
        print(brief.cross_article_synthesis)

    if warnings:
        print(f"\n{line}")
        print(f"VALIDATION WARNINGS ({len(warnings)})")
        print(line)
        for warning in warnings:
            print(f"  ! {warning}")

    print()


def render_markdown(brief: ResearchBrief, warnings: list[str]) -> str:
    """Render the verified brief as a Markdown document."""
    timestamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [
        "# Hacker News Research Brief",
        "",
        f"_Generated on {timestamp}_",
        "",
        "## Per-article insights",
        "",
    ]

    for index, insight in enumerate(brief.insights, start=1):
        lines.append(f"### {index}. {insight.title}")
        lines.append("")
        lines.append(f"- **Source ID:** `{insight.source_id}`")
        lines.append(f"- **URL:** <{insight.url}>")
        lines.append(f"- **Sentiment:** {insight.sentiment}")
        lines.append("")
        if insight.summary:
            lines.append(insight.summary)
            lines.append("")
        if insight.key_points:
            lines.append("**Key points**")
            lines.append("")
            lines.extend(f"- {point}" for point in insight.key_points)
            lines.append("")
        if insight.evidence:
            lines.append("**Evidence**")
            lines.append("")
            lines.extend(f"> {quote}" for quote in insight.evidence)
            lines.append("")

    if brief.overall_summary:
        lines.extend(["## Overall summary", "", brief.overall_summary, ""])

    if brief.cross_article_synthesis:
        lines.extend(
            ["## Cross-article synthesis", "", brief.cross_article_synthesis, ""]
        )

    if warnings:
        lines.extend([f"## Validation warnings ({len(warnings)})", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")

    return "\n".join(lines)


def save_markdown(markdown: str) -> str:
    """Write the Markdown brief to a timestamped file in ``output/``."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"research_brief_{timestamp}.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(markdown)
    return path


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main() -> int:
    """Run the full Research Brief pipeline end to end."""
    load_dotenv()
    model = os.getenv("OPENAI_MODEL") or DEFAULT_MODEL

    try:
        print("[1/6] Fetching Hacker News front page...")
        html = fetch_frontpage()
        metadata = parse_frontpage(html)
        print(f"      Parsed {len(metadata)} stories.")

        print("[2/6] Filtering external candidates...")
        candidates = filter_external(metadata)
        seen_ids = load_seen_ids()
        fresh_candidates = filter_unseen(candidates, seen_ids)
        print(
            f"      {len(candidates)} external candidates "
            f"({len(candidates) - len(fresh_candidates)} already seen, "
            f"{len(fresh_candidates)} new)."
        )

        print(f"[3/6] Extracting article content (up to {MAX_ARTICLES})...")
        articles = extract_articles(fresh_candidates, MAX_ARTICLES)
        print(f"      Extracted {len(articles)} usable articles.")

        print("[4/6] Validating article records...")
        known_ids = {meta.source_id for meta in metadata}
        articles, record_warnings = validate_articles(articles, known_ids)
        if not articles:
            print("      No new usable articles remain. Aborting.")
            return 1

        print(f"[5/6] Analysing articles with model '{model}'...")
        client = build_client()
        try:
            brief = analyze_articles(articles, client, model)
        except Exception as exc:  # noqa: BLE001 - resilient model fallback
            if model != DEFAULT_MODEL:
                print(
                    f"      ! Model '{model}' failed ({exc}); "
                    f"retrying with '{DEFAULT_MODEL}'."
                )
                brief = analyze_articles(articles, client, DEFAULT_MODEL)
            else:
                raise
        brief, app_warnings = verify_brief(brief, articles)

        print("[6/6] Rendering report...")
        warnings = record_warnings + app_warnings
        render_terminal_report(brief, warnings)

        markdown = render_markdown(brief, warnings)
        path = save_markdown(markdown)
        print(f"Markdown artifact saved to: {path}")

        # Remember the freshly processed articles so future runs skip them.
        seen_ids |= {article.source_id for article in articles}
        try:
            save_seen_ids(seen_ids)
            print(f"Deduplication state updated ({len(seen_ids)} remembered).")
        except OSError as exc:
            print(f"      ! Could not persist deduplication state: {exc}")

        return 0

    except (requests.RequestException, RuntimeError, ValidationError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
