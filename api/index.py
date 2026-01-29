"""Main FastAPI application for Vercel deployment."""

import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional
import xml.etree.ElementTree as ET

import httpx
import feedparser
import yaml
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Template

# === Configuration ===

@dataclass
class ResearchArea:
    name: str
    terms: list[str]
    weight: float = 1.0

@dataclass
class Config:
    areas: dict[str, ResearchArea]
    lookback_days: int
    title_multiplier: float
    pubmed_enabled: bool
    biorxiv_enabled: bool
    arxiv_enabled: bool
    arxiv_categories: list[str]

def load_config() -> Config:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    areas = {}
    for name, area_raw in raw.get("areas", {}).items():
        areas[name] = ResearchArea(
            name=name,
            terms=area_raw.get("terms", []),
            weight=area_raw.get("weight", 1.0),
        )

    sources = raw.get("sources", {})
    arxiv = sources.get("arxiv", {})

    return Config(
        areas=areas,
        lookback_days=raw.get("collection", {}).get("lookback_days", 7),
        title_multiplier=raw.get("scoring", {}).get("title_multiplier", 2.0),
        pubmed_enabled=sources.get("pubmed", True),
        biorxiv_enabled=sources.get("biorxiv", True),
        arxiv_enabled=arxiv.get("enabled", True) if isinstance(arxiv, dict) else arxiv,
        arxiv_categories=arxiv.get("categories", ["cs.AI", "q-bio.GN"]) if isinstance(arxiv, dict) else ["cs.AI", "q-bio.GN"],
    )

# === Paper Model ===

@dataclass
class Paper:
    title: str
    authors: str
    abstract: str
    url: str
    published_date: date
    source: str
    matched_areas: list[str] = field(default_factory=list)
    score: float = 0.0

# === Collectors ===

async def fetch_pubmed(days: int, client: httpx.AsyncClient) -> list[Paper]:
    """Fetch papers from PubMed."""
    papers = []
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    date_range = f"{start_date:%Y/%m/%d}:{end_date:%Y/%m/%d}[edat]"

    try:
        # Search for IDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "term": date_range, "retmax": 200, "retmode": "json", "sort": "pub_date"}
        resp = await client.get(search_url, params=params)
        resp.raise_for_status()
        ids = resp.json().get("esearchresult", {}).get("idlist", [])

        if not ids:
            return papers

        # Fetch details
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {"db": "pubmed", "id": ",".join(ids[:100]), "retmode": "xml"}
        resp = await client.get(fetch_url, params=params)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        for article in root.findall(".//PubmedArticle"):
            try:
                paper = _parse_pubmed_article(article)
                if paper:
                    papers.append(paper)
            except Exception:
                continue
    except Exception as e:
        print(f"PubMed error: {e}")

    return papers

def _parse_pubmed_article(article: ET.Element) -> Optional[Paper]:
    medline = article.find(".//MedlineCitation")
    if medline is None:
        return None

    article_elem = medline.find(".//Article")
    if article_elem is None:
        return None

    title_elem = article_elem.find(".//ArticleTitle")
    title = title_elem.text if title_elem is not None and title_elem.text else ""
    if not title:
        return None

    abstract_parts = []
    for abs_text in article_elem.findall(".//Abstract/AbstractText"):
        text = abs_text.text or ""
        abstract_parts.append(text)
    abstract = " ".join(abstract_parts)

    authors = []
    for author in article_elem.findall(".//AuthorList/Author"):
        last = author.find("LastName")
        if last is not None and last.text:
            authors.append(last.text)
    authors_str = ", ".join(authors[:5])
    if len(authors) > 5:
        authors_str += " et al."

    pmid_elem = medline.find(".//PMID")
    pmid = pmid_elem.text if pmid_elem is not None else ""
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

    return Paper(
        title=title,
        authors=authors_str,
        abstract=abstract,
        url=url,
        published_date=date.today(),
        source="pubmed",
    )

async def fetch_biorxiv(days: int, client: httpx.AsyncClient) -> list[Paper]:
    """Fetch papers from bioRxiv."""
    papers = []
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    try:
        url = f"https://api.biorxiv.org/details/biorxiv/{start_date}/{end_date}/0"
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("collection", [])[:100]:
            try:
                doi = item.get("doi", "")
                papers.append(Paper(
                    title=item.get("title", "").strip(),
                    authors=item.get("authors", ""),
                    abstract=item.get("abstract", ""),
                    url=f"https://www.biorxiv.org/content/{doi}" if doi else "",
                    published_date=date.fromisoformat(item.get("date", str(date.today()))),
                    source="biorxiv",
                ))
            except Exception:
                continue
    except Exception as e:
        print(f"bioRxiv error: {e}")

    return papers

async def fetch_arxiv(days: int, categories: list[str], client: httpx.AsyncClient) -> list[Paper]:
    """Fetch papers from arXiv."""
    papers = []

    try:
        cat_query = " OR ".join(f"cat:{cat}" for cat in categories)
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"({cat_query})",
            "start": 0,
            "max_results": 100,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        resp = await client.get(url, params=params)
        resp.raise_for_status()

        feed = feedparser.parse(resp.text)
        cutoff = date.today() - timedelta(days=days)

        for entry in feed.entries:
            try:
                published = entry.get("published", "")
                dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                pub_date = dt.date()

                if pub_date < cutoff:
                    continue

                authors = [a.get("name", "") for a in entry.get("authors", [])]
                authors_str = ", ".join(authors[:5])
                if len(authors) > 5:
                    authors_str += " et al."

                papers.append(Paper(
                    title=entry.get("title", "").replace("\n", " ").strip(),
                    authors=authors_str,
                    abstract=entry.get("summary", "").replace("\n", " ").strip(),
                    url=entry.get("link", ""),
                    published_date=pub_date,
                    source="arxiv",
                ))
            except Exception:
                continue
    except Exception as e:
        print(f"arXiv error: {e}")

    return papers

# === Scoring ===

def score_paper(paper: Paper, config: Config) -> Paper:
    """Score a paper based on keyword matching."""
    title = paper.title.lower()
    abstract = paper.abstract.lower()

    matched_areas = []
    total_score = 0.0

    for area_name, area in config.areas.items():
        area_score = 0.0
        for term in area.terms:
            term_lower = term.lower()
            if term_lower in title:
                area_score += config.title_multiplier
            if term_lower in abstract:
                area_score += 1.0

        if area_score > 0:
            matched_areas.append(area_name)
            total_score += area_score * area.weight

    paper.matched_areas = matched_areas
    paper.score = round(total_score, 1)
    return paper

# === Main Fetch Function ===

async def fetch_all_papers(config: Config) -> list[Paper]:
    """Fetch papers from all sources and score them."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = []

        if config.pubmed_enabled:
            tasks.append(fetch_pubmed(config.lookback_days, client))
        if config.biorxiv_enabled:
            tasks.append(fetch_biorxiv(config.lookback_days, client))
        if config.arxiv_enabled:
            tasks.append(fetch_arxiv(config.lookback_days, config.arxiv_categories, client))

        results = await asyncio.gather(*tasks, return_exceptions=True)

    all_papers = []
    for result in results:
        if isinstance(result, list):
            all_papers.extend(result)

    # Score and filter
    scored = [score_paper(p, config) for p in all_papers]
    relevant = [p for p in scored if p.score > 0]

    # Sort by score descending
    relevant.sort(key=lambda p: (-p.score, p.published_date), reverse=False)

    return relevant

# === FastAPI App ===

app = FastAPI(title="PaperFeed")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PaperFeed</title>
    <style>
        :root {
            --bg: #fafafa;
            --card: #ffffff;
            --text: #1a1a1a;
            --muted: #666;
            --border: #e0e0e0;
            --accent: #2563eb;
            --score: #16a34a;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.5;
            font-size: 14px;
        }
        .container { max-width: 900px; margin: 0 auto; padding: 1rem; }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 1rem;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        header h1 { font-size: 1.5rem; font-weight: 600; }
        .stats { font-size: 0.85rem; color: var(--muted); }
        .controls {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-bottom: 1rem;
            align-items: center;
        }
        select, button {
            padding: 0.5rem 0.75rem;
            border: 1px solid var(--border);
            border-radius: 4px;
            font-size: 0.85rem;
            background: var(--card);
        }
        button {
            background: var(--accent);
            color: white;
            border: none;
            cursor: pointer;
        }
        button:hover { opacity: 0.9; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .loading { display: none; }
        .papers { display: flex; flex-direction: column; gap: 0.75rem; }
        .paper {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 1rem;
        }
        .paper-title {
            font-size: 1rem;
            font-weight: 500;
            color: var(--accent);
            text-decoration: none;
            display: block;
            margin-bottom: 0.5rem;
        }
        .paper-title:hover { text-decoration: underline; }
        .paper-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            font-size: 0.8rem;
            color: var(--muted);
            margin-bottom: 0.5rem;
        }
        .score {
            background: var(--score);
            color: white;
            padding: 0.1rem 0.4rem;
            border-radius: 3px;
            font-weight: 600;
        }
        .areas { display: flex; flex-wrap: wrap; gap: 0.25rem; margin-bottom: 0.5rem; }
        .area-tag {
            padding: 0.15rem 0.4rem;
            background: var(--accent);
            color: white;
            border-radius: 3px;
            font-size: 0.7rem;
        }
        .abstract {
            font-size: 0.85rem;
            color: var(--muted);
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        .empty { text-align: center; padding: 3rem; color: var(--muted); }
        .source-badge {
            padding: 0.1rem 0.3rem;
            background: var(--border);
            border-radius: 3px;
            font-size: 0.7rem;
            text-transform: uppercase;
        }
        @media (max-width: 600px) {
            .controls { flex-direction: column; align-items: stretch; }
            select, button { width: 100%; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>PaperFeed</h1>
            <div class="stats">{{ papers|length }} papers found</div>
        </header>

        <form class="controls" method="get">
            <select name="area">
                <option value="">All areas</option>
                {% for a in areas %}
                <option value="{{ a }}" {% if area == a %}selected{% endif %}>{{ a }}</option>
                {% endfor %}
            </select>
            <select name="source">
                <option value="">All sources</option>
                <option value="pubmed" {% if source == 'pubmed' %}selected{% endif %}>PubMed</option>
                <option value="biorxiv" {% if source == 'biorxiv' %}selected{% endif %}>bioRxiv</option>
                <option value="arxiv" {% if source == 'arxiv' %}selected{% endif %}>arXiv</option>
            </select>
            <select name="sort">
                <option value="score" {% if sort == 'score' %}selected{% endif %}>Sort by relevance</option>
                <option value="date" {% if sort == 'date' %}selected{% endif %}>Sort by date</option>
            </select>
            <button type="submit">Apply Filters</button>
        </form>

        <div class="papers">
            {% if papers %}
                {% for paper in papers %}
                <article class="paper">
                    <a href="{{ paper.url }}" target="_blank" rel="noopener" class="paper-title">
                        {{ paper.title }}
                    </a>
                    <div class="paper-meta">
                        <span class="score">{{ paper.score }}</span>
                        <span class="source-badge">{{ paper.source }}</span>
                        <span>{{ paper.authors }}</span>
                        <span>{{ paper.published_date }}</span>
                    </div>
                    {% if paper.matched_areas %}
                    <div class="areas">
                        {% for a in paper.matched_areas %}
                        <span class="area-tag">{{ a }}</span>
                        {% endfor %}
                    </div>
                    {% endif %}
                    <p class="abstract">{{ paper.abstract }}</p>
                </article>
                {% endfor %}
            {% else %}
                <div class="empty">
                    <p>No papers found matching your criteria.</p>
                    <p>Try adjusting filters or check back later.</p>
                </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    area: str = Query(""),
    source: str = Query(""),
    sort: str = Query("score"),
):
    config = load_config()
    papers = await fetch_all_papers(config)

    # Filter by area
    if area:
        papers = [p for p in papers if area in p.matched_areas]

    # Filter by source
    if source:
        papers = [p for p in papers if p.source == source]

    # Sort
    if sort == "date":
        papers.sort(key=lambda p: p.published_date, reverse=True)
    else:
        papers.sort(key=lambda p: p.score, reverse=True)

    template = Template(HTML_TEMPLATE)
    html = template.render(
        papers=papers,
        areas=list(config.areas.keys()),
        area=area,
        source=source,
        sort=sort,
    )
    return HTMLResponse(html)

@app.get("/api/health")
async def health():
    return {"status": "ok"}
