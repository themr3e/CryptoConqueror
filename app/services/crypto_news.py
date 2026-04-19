"""Crypto news fetcher for trading context.

Fetches recent headlines from two sources (no API key required):
  1. CryptoPanic free tier — if CRYPTOPANIC_API_KEY is set
  2. Binance announcements RSS — always available, no auth

Headlines are filtered for relevance to the symbols being traded
and injected into Claude's market context before each decision.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import httpx
from loguru import logger

from app.config import get_settings

# Symbol → search keywords mapping
_SYMBOL_KEYWORDS: dict[str, list[str]] = {
    "BTCUSDT":    ["bitcoin", "btc"],
    "ETHUSDT":    ["ethereum", "eth"],
    "SOLUSDT":    ["solana", "sol"],
    "BNBUSDT":    ["bnb", "binance coin"],
    "XRPUSDT":    ["xrp", "ripple"],
    "ADAUSDT":    ["cardano", "ada"],
    "DOGEUSDT":   ["dogecoin", "doge"],
    "AVAXUSDT":   ["avalanche", "avax"],
    "DOTUSDT":    ["polkadot", "dot"],
    "LINKUSDT":   ["chainlink", "link"],
    "ARBUSDT":    ["arbitrum", "arb"],
    "OPUSDT":     ["optimism", "op"],
    "INJUSDT":    ["injective", "inj"],
    "SUIUSDT":    ["sui"],
    "SEIUSDT":    ["sei"],
    "TIAUSDT":    ["celestia", "tia"],
    "JUPUSDT":    ["jupiter", "jup"],
    "PYTHUSDT":   ["pyth"],
    "WLDUSDT":    ["worldcoin", "wld"],
    "RENDERUSDT": ["render", "rndr"],
}

_BINANCE_RSS = "https://www.binance.com/en/support/announcement/rss.xml"
_CRYPTOPANIC_URL = "https://cryptopanic.com/api/free/v1/posts/?auth_token={token}&kind=news&filter=hot&public=true"


class NewsItem:
    def __init__(self, title: str, published: datetime | None, source: str) -> None:
        self.title = title
        self.published = published
        self.source = source

    def age_str(self) -> str:
        if self.published is None:
            return ""
        now = datetime.now(timezone.utc)
        delta = now - self.published
        if delta < timedelta(hours=1):
            return f"{int(delta.total_seconds()/60)}m ago"
        if delta < timedelta(hours=24):
            return f"{int(delta.total_seconds()/3600)}h ago"
        return f"{delta.days}d ago"


async def fetch_news(symbols: list[str], max_items: int = 8) -> str:
    """Return a formatted news block for Claude's context.

    Fetches from CryptoPanic (if key set) then Binance announcements.
    Returns empty string if no news is available.
    """
    items: list[NewsItem] = []

    settings = get_settings()
    token = getattr(settings, "cryptopanic_api_key", "")

    if token:
        items.extend(await _fetch_cryptopanic(token))
    items.extend(await _fetch_binance_rss())

    if not items:
        return ""

    # Filter for recent items (last 24h) and general crypto/market keywords
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    relevant_keywords = _build_keyword_set(symbols)

    scored: list[tuple[int, NewsItem]] = []
    for item in items:
        if item.published and item.published < cutoff:
            continue
        title_lower = item.title.lower()
        # Score: how many relevant keywords appear in title
        score = sum(1 for kw in relevant_keywords if kw in title_lower)
        # Always include items with high-impact words regardless of symbol match
        if any(w in title_lower for w in ["fed", "sec", "etf", "ban", "hack", "crash",
                                           "surge", "rally", "dump", "listing", "halving",
                                           "rate", "inflation", "regulation"]):
            score += 2
        if score > 0 or any(w in title_lower for w in ["crypto", "bitcoin", "market"]):
            scored.append((score, item))

    # Sort by score desc, then by recency
    scored.sort(key=lambda x: (x[0], x[1].published or datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
    top = [item for _, item in scored[:max_items]]

    if not top:
        return ""

    lines = ["📰 [CRYPTO NEWS — last 24h]"]
    for item in top:
        age = item.age_str()
        age_str = f" ({age})" if age else ""
        lines.append(f"  • {item.title}{age_str}")
    lines.append("  Use news as a FILTER: avoid trading against major news momentum.")
    return "\n".join(lines)


# ── Fetchers ──────────────────────────────────────────────────────────────────

async def _fetch_cryptopanic(token: str) -> list[NewsItem]:
    url = _CRYPTOPANIC_URL.format(token=token)
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return []
            data = resp.json()
            items = []
            for post in data.get("results", [])[:20]:
                title = post.get("title", "")
                published_at = post.get("published_at", "")
                try:
                    pub = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                except Exception:
                    pub = None
                items.append(NewsItem(title=title, published=pub, source="CryptoPanic"))
            return items
    except Exception as exc:
        logger.debug("[CryptoNews] CryptoPanic fetch failed: {}", exc)
        return []


async def _fetch_binance_rss() -> list[NewsItem]:
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(_BINANCE_RSS)
            if resp.status_code != 200:
                return []
            root = ET.fromstring(resp.text)
            channel = root.find("channel")
            if channel is None:
                return []
            items = []
            for entry in channel.findall("item")[:15]:
                title_el = entry.find("title")
                pub_el   = entry.find("pubDate")
                title = title_el.text.strip() if title_el is not None and title_el.text else ""
                pub = None
                if pub_el is not None and pub_el.text:
                    try:
                        pub = parsedate_to_datetime(pub_el.text)
                        if pub.tzinfo is None:
                            pub = pub.replace(tzinfo=timezone.utc)
                    except Exception:
                        pass
                if title:
                    items.append(NewsItem(title=title, published=pub, source="Binance"))
            return items
    except Exception as exc:
        logger.debug("[CryptoNews] Binance RSS fetch failed: {}", exc)
        return []


def _build_keyword_set(symbols: list[str]) -> set[str]:
    keywords: set[str] = set()
    for sym in symbols:
        kws = _SYMBOL_KEYWORDS.get(sym.upper(), [])
        keywords.update(kws)
    # Always include generic crypto terms
    keywords.update(["crypto", "bitcoin", "ethereum", "defi", "altcoin", "market"])
    return keywords
