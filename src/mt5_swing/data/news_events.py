"""News / geopolitics intensity feeds for scholarly FX event studies.

Design
------
Prefer free public intensity series when available:

1. **GDELT DOC 2.0** ``TimelineVol`` for conflict/geopolitics query volume
   (https://api.gdeltproject.org/api/v2/doc/doc). Free, no key. Practical limits:
   recent window only (~3 months for reliable timeline), frequent HTTP 429.
2. **Yahoo Finance / Reuters RSS** daily headline *counts* (ethical public feeds,
   not HTML scraping). Useful for recent monitoring; **not** a multi-year panel.

If live feeds are blocked, rate-limited, or too short for an event study, fall
back to **Caldara–Iacoviello daily GPR spikes** as a news-based geopolitical
intensity *proxy* (same newspaper-count foundation as academic GPR). Document
the limitation: GPR is not raw NLP of today's headlines, and publication lag
must be respected.

Paid APIs (Refinitiv, Bloomberg, RavenPack, GDELT Cloud with key, etc.) would
be needed for full-history multilingual NLP sentiment/entity panels — not wired
here.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from mt5_swing.data.macro_uncertainty import load_gpr, macro_dir

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
# Conflict / geopolitics query (DOC operators). Kept simple to reduce 429s.
GDELT_DEFAULT_QUERY = '(war OR conflict OR "geopolitical risk" OR invasion OR missile)'
YAHOO_FX_RSS = "https://finance.yahoo.com/rss/headline?s=EURUSD=X"
REUTERS_WORLD_RSS = "https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best"

FeedSource = Literal["gdelt_doc", "yahoo_rss", "reuters_rss", "gpr_spike_proxy", "stub"]


@dataclass
class NewsIntensityMeta:
    source: FeedSource
    available: bool
    note: str
    pub_lag_days: int = 1
    n_obs: int = 0
    query: str = ""
    paid_nlp_needed: bool = True  # honest default for full NLP history


class NewsIntensityFeed(Protocol):
    """Minimal interface: daily intensity series + metadata."""

    def fetch(self, *, force: bool = False) -> pd.Series: ...

    @property
    def meta(self) -> NewsIntensityMeta: ...


def _ensure_macro() -> Path:
    d = macro_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _http_get(url: str, *, timeout: float = 30.0, retries: int = 2) -> bytes:
    last: Exception | None = None
    for i in range(retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "mt5-swing-scholarly-research/1.0"})
            with urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last = exc
            if isinstance(exc, HTTPError) and exc.code == 429:
                time.sleep(2.0 * (i + 1))
                continue
            time.sleep(0.5 * (i + 1))
    assert last is not None
    raise last


# ---------------------------------------------------------------------------
# GDELT DOC TimelineVol
# ---------------------------------------------------------------------------


@dataclass
class GdeltDocFeed:
    """GDELT DOC 2.0 article-volume timeline for geopolitics keywords."""

    query: str = GDELT_DEFAULT_QUERY
    timespan: str = "3m"  # API-friendly recent window
    cache_name: str = "gdelt_conflict_timeline.csv"
    pub_lag_days: int = 1
    _meta: NewsIntensityMeta = field(init=False)

    def __post_init__(self) -> None:
        self._meta = NewsIntensityMeta(
            source="gdelt_doc",
            available=False,
            note="not fetched yet",
            pub_lag_days=self.pub_lag_days,
            query=self.query,
            paid_nlp_needed=True,
        )

    @property
    def meta(self) -> NewsIntensityMeta:
        return self._meta

    def cache_path(self) -> Path:
        return _ensure_macro() / self.cache_name

    def fetch(self, *, force: bool = False) -> pd.Series:
        path = self.cache_path()
        if path.exists() and not force and path.stat().st_size > 50:
            s = self._read_cache(path)
            self._meta = NewsIntensityMeta(
                source="gdelt_doc",
                available=True,
                note=f"cached CSV {path.name} (DOC TimelineVol; ~recent window only)",
                pub_lag_days=self.pub_lag_days,
                n_obs=int(s.notna().sum()),
                query=self.query,
                paid_nlp_needed=True,
            )
            return self._apply_lag(s)

        params = {
            "query": self.query,
            "mode": "TimelineVol",
            "timespan": self.timespan,
            "format": "json",
        }
        url = f"{GDELT_DOC_API}?{urlencode(params)}"
        try:
            raw = _http_get(url, timeout=45.0, retries=2)
            payload = json.loads(raw.decode("utf-8", errors="replace"))
            s = self._parse_timeline(payload)
            if s.empty:
                raise RuntimeError("GDELT TimelineVol returned empty series")
            out = s.rename("intensity").to_frame()
            out.index.name = "date"
            out.to_csv(path)
            self._meta = NewsIntensityMeta(
                source="gdelt_doc",
                available=True,
                note=f"live DOC TimelineVol timespan={self.timespan}",
                pub_lag_days=self.pub_lag_days,
                n_obs=int(s.notna().sum()),
                query=self.query,
                paid_nlp_needed=True,
            )
            return self._apply_lag(s)
        except Exception as exc:  # noqa: BLE001 — feed fallback is intentional
            self._meta = NewsIntensityMeta(
                source="gdelt_doc",
                available=False,
                note=f"GDELT blocked/failed ({type(exc).__name__}: {exc}). "
                "Use GPR spike proxy for multi-year event studies. "
                "DOC TimelineVol is also ~3-month limited even when healthy.",
                pub_lag_days=self.pub_lag_days,
                n_obs=0,
                query=self.query,
                paid_nlp_needed=True,
            )
            raise RuntimeError(self._meta.note) from exc

    def _parse_timeline(self, payload: object) -> pd.Series:
        # DOC JSON shapes vary: {"timeline":[{"data":[{"date":..., "value":...}]}]}
        # or list of {datetime, value} / {date, count}
        rows: list[tuple[pd.Timestamp, float]] = []
        if isinstance(payload, dict):
            timeline = payload.get("timeline") or payload.get("data") or []
            if isinstance(timeline, list) and timeline and isinstance(timeline[0], dict):
                # nested data
                for block in timeline:
                    data = block.get("data") if isinstance(block, dict) else None
                    if isinstance(data, list):
                        for pt in data:
                            self._append_point(rows, pt)
                    else:
                        self._append_point(rows, block)
            elif "date" in payload or "datetime" in payload:
                self._append_point(rows, payload)
        elif isinstance(payload, list):
            for pt in payload:
                self._append_point(rows, pt)
        if not rows:
            return pd.Series(dtype=float, name="intensity")
        idx = pd.DatetimeIndex([r[0] for r in rows], tz="UTC")
        s = pd.Series([r[1] for r in rows], index=idx, name="intensity")
        return s.groupby(level=0).mean().sort_index()

    @staticmethod
    def _append_point(rows: list[tuple[pd.Timestamp, float]], pt: object) -> None:
        if not isinstance(pt, dict):
            return
        date_raw = pt.get("date") or pt.get("datetime") or pt.get("seenDate")
        val_raw = pt.get("value") or pt.get("count") or pt.get("intensity") or pt.get("Article Count")
        if date_raw is None or val_raw is None:
            return
        try:
            ts = pd.to_datetime(date_raw, utc=True)
            val = float(val_raw)
        except (TypeError, ValueError):
            return
        rows.append((ts, val))

    def _read_cache(self, path: Path) -> pd.Series:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        s.index = pd.to_datetime(s.index, utc=True)
        s.name = "intensity"
        return s.dropna().sort_index()

    def _apply_lag(self, s: pd.Series) -> pd.Series:
        out = s.copy()
        if self.pub_lag_days > 0:
            out.index = out.index + pd.Timedelta(days=int(self.pub_lag_days))
            out = out[~out.index.duplicated(keep="last")].sort_index()
        out.attrs["source"] = "gdelt_doc"
        out.attrs["pub_lag_days"] = int(self.pub_lag_days)
        out.attrs["query"] = self.query
        return out


# ---------------------------------------------------------------------------
# Ethical RSS headline counts (recent only)
# ---------------------------------------------------------------------------


@dataclass
class RssHeadlineCountFeed:
    """Count headlines from a public RSS URL (ethical; not HTML scrape).

    Returns at most a short recent window (feed contents), useful as a live
    monitor — **not** sufficient for multi-year event studies.
    """

    url: str = YAHOO_FX_RSS
    source_name: FeedSource = "yahoo_rss"
    cache_name: str = "yahoo_fx_rss_counts.csv"
    pub_lag_days: int = 1
    _meta: NewsIntensityMeta = field(init=False)

    def __post_init__(self) -> None:
        self._meta = NewsIntensityMeta(
            source=self.source_name,
            available=False,
            note="not fetched yet",
            pub_lag_days=self.pub_lag_days,
            paid_nlp_needed=True,
        )

    @property
    def meta(self) -> NewsIntensityMeta:
        return self._meta

    def fetch(self, *, force: bool = False) -> pd.Series:
        path = _ensure_macro() / self.cache_name
        try:
            raw = _http_get(self.url, timeout=20.0, retries=1)
            text = raw.decode("utf-8", errors="replace")
            # Minimal date parse from <pubDate> without requiring feedparser
            import re
            from email.utils import parsedate_to_datetime

            dates = []
            for m in re.finditer(r"<pubDate>([^<]+)</pubDate>", text, flags=re.I):
                try:
                    dt = parsedate_to_datetime(m.group(1).strip())
                    if dt.tzinfo is None:
                        from datetime import timezone
                        dt = dt.replace(tzinfo=timezone.utc)
                    dates.append(pd.Timestamp(dt).tz_convert("UTC").normalize())
                except (TypeError, ValueError, IndexError):
                    continue
            if not dates:
                raise RuntimeError("no pubDate entries in RSS")
            counts = pd.Series(1, index=pd.DatetimeIndex(dates)).groupby(level=0).sum()
            counts = counts.sort_index().astype(float)
            counts.name = "intensity"
            counts.to_frame().to_csv(path)
            self._meta = NewsIntensityMeta(
                source=self.source_name,
                available=True,
                note=f"RSS headline counts from {self.url} (recent feed items only)",
                pub_lag_days=self.pub_lag_days,
                n_obs=int(len(counts)),
                paid_nlp_needed=True,
            )
            if self.pub_lag_days > 0:
                counts.index = counts.index + pd.Timedelta(days=int(self.pub_lag_days))
                counts = counts[~counts.index.duplicated(keep="last")].sort_index()
            counts.attrs["source"] = self.source_name
            return counts
        except Exception as exc:  # noqa: BLE001
            self._meta = NewsIntensityMeta(
                source=self.source_name,
                available=False,
                note=f"RSS fetch failed ({type(exc).__name__}: {exc})",
                pub_lag_days=self.pub_lag_days,
                n_obs=0,
                paid_nlp_needed=True,
            )
            raise RuntimeError(self._meta.note) from exc


# ---------------------------------------------------------------------------
# GPR daily spike proxy (multi-year)
# ---------------------------------------------------------------------------


@dataclass
class GprSpikeProxyFeed:
    """Use lagged daily Caldara–Iacoviello GPR as geopolitics news intensity."""

    pub_lag_days: int = 1
    column: str = "GPR"
    _meta: NewsIntensityMeta = field(init=False)

    def __post_init__(self) -> None:
        self._meta = NewsIntensityMeta(
            source="gpr_spike_proxy",
            available=False,
            note="not loaded yet",
            pub_lag_days=self.pub_lag_days,
            paid_nlp_needed=True,
        )

    @property
    def meta(self) -> NewsIntensityMeta:
        return self._meta

    def fetch(self, *, force: bool = False) -> pd.Series:
        s = load_gpr(
            freq="D",
            column=self.column,
            download=True,
            pub_lag_days=self.pub_lag_days,
            force=force,
        )
        s = s.rename("intensity")
        self._meta = NewsIntensityMeta(
            source="gpr_spike_proxy",
            available=True,
            note=(
                "Proxy: Caldara–Iacoviello daily GPR (newspaper article share). "
                "Not live NLP of Reuters/Yahoo headlines; multi-year coverage. "
                "Limitation: intensity ≠ signed sentiment / entity NLP."
            ),
            pub_lag_days=self.pub_lag_days,
            n_obs=int(s.notna().sum()),
            paid_nlp_needed=True,
        )
        s.attrs["source"] = "gpr_spike_proxy"
        s.attrs["pub_lag_days"] = int(self.pub_lag_days)
        s.attrs["limitation"] = self._meta.note
        return s


@dataclass
class StubNewsFeed:
    """Empty feed documenting missing NLP panel."""

    note: str = "Stub: no live news intensity. Prefer GDELT or GPR proxy."

    @property
    def meta(self) -> NewsIntensityMeta:
        return NewsIntensityMeta(
            source="stub",
            available=False,
            note=self.note,
            paid_nlp_needed=True,
        )

    def fetch(self, *, force: bool = False) -> pd.Series:
        raise RuntimeError(self.note)


@dataclass
class ResolvedNewsIntensity:
    series: pd.Series
    meta: NewsIntensityMeta
    attempted: list[NewsIntensityMeta] = field(default_factory=list)


def resolve_news_intensity(
    *,
    prefer_gdelt: bool = True,
    try_rss: bool = False,
    force: bool = False,
    min_obs_for_gdelt: int = 40,
    pub_lag_days: int = 1,
) -> ResolvedNewsIntensity:
    """Prefer GDELT (if usable length), else GPR daily spike proxy.

    RSS is optional and never used alone for multi-year studies.
    """
    attempted: list[NewsIntensityMeta] = []

    if prefer_gdelt:
        gdelt = GdeltDocFeed(pub_lag_days=pub_lag_days)
        try:
            s = gdelt.fetch(force=force)
            attempted.append(gdelt.meta)
            if int(s.notna().sum()) >= min_obs_for_gdelt:
                # Still too short for year/holdout event study — mark and fall through
                # unless caller wants recent-only. We require ~1y for scholarly board.
                if int(s.notna().sum()) >= 200:
                    return ResolvedNewsIntensity(series=s, meta=gdelt.meta, attempted=attempted)
                # Keep attempt note; fall back to GPR for history
                short = NewsIntensityMeta(
                    source="gdelt_doc",
                    available=True,
                    note=gdelt.meta.note + f" — only n={s.notna().sum()} days; too short for multi-year study → GPR proxy",
                    pub_lag_days=pub_lag_days,
                    n_obs=int(s.notna().sum()),
                    query=gdelt.query,
                    paid_nlp_needed=True,
                )
                attempted.append(short)
            else:
                attempted.append(gdelt.meta)
        except Exception:  # noqa: BLE001
            attempted.append(gdelt.meta)

    if try_rss:
        for src, url, cache in (
            ("yahoo_rss", YAHOO_FX_RSS, "yahoo_fx_rss_counts.csv"),
            ("reuters_rss", REUTERS_WORLD_RSS, "reuters_world_rss_counts.csv"),
        ):
            feed = RssHeadlineCountFeed(url=url, source_name=src, cache_name=cache, pub_lag_days=pub_lag_days)  # type: ignore[arg-type]
            try:
                feed.fetch(force=force)
            except Exception:  # noqa: BLE001
                pass
            attempted.append(feed.meta)

    proxy = GprSpikeProxyFeed(pub_lag_days=pub_lag_days)
    s = proxy.fetch(force=force)
    attempted.append(proxy.meta)
    return ResolvedNewsIntensity(series=s, meta=proxy.meta, attempted=attempted)


def high_intensity_event_dates(
    intensity: pd.Series,
    index: pd.DatetimeIndex,
    *,
    decile: float = 0.90,
    min_gap_days: int = 5,
    extra_lag: int = 1,
) -> tuple[pd.DatetimeIndex, dict]:
    """Event dates = top-``decile`` of lagged intensity on ``index``, thinned."""
    from mt5_swing.strategies.gpr_regime import align_macro_to_index

    g = align_macro_to_index(intensity, index)
    g_lag = g.shift(int(extra_lag))
    thr = g_lag.quantile(decile)
    events = g_lag[g_lag >= thr].dropna().index
    kept: list[pd.Timestamp] = []
    last: pd.Timestamp | None = None
    for dt in events:
        if last is None or (dt - last).days >= int(min_gap_days):
            kept.append(dt)
            last = dt
    out = pd.DatetimeIndex(kept)
    # Store meta on a Series wrapper-friendly dict via module-level side channel
    # (DatetimeIndex.attrs is not reliably available across pandas versions).
    meta = {
        "threshold": float(thr) if np.isfinite(thr) else float("nan"),
        "decile": float(decile),
        "n_raw": int(len(events)),
    }
    return out, meta


def control_dates(
    index: pd.DatetimeIndex,
    events: pd.DatetimeIndex,
    *,
    n: int | None = None,
    seed: int = 42,
    exclude_window: int = 5,
) -> pd.DatetimeIndex:
    """Random non-event trading days, excluding ±exclude_window around events."""
    idx = pd.DatetimeIndex(index)
    ban = set()
    pos = {dt: i for i, dt in enumerate(idx)}
    for e in events:
        if e not in pos:
            # nearest pad
            loc = idx.get_indexer([e], method="pad")[0]
            if loc < 0:
                continue
            e = idx[loc]
        i = pos[e]
        for j in range(max(0, i - exclude_window), min(len(idx), i + exclude_window + 1)):
            ban.add(idx[j])
    pool = [dt for dt in idx if dt not in ban]
    n = int(n if n is not None else len(events))
    if n <= 0 or not pool:
        return pd.DatetimeIndex([])
    rng = np.random.default_rng(seed)
    take = min(n, len(pool))
    chosen = rng.choice(np.array(pool, dtype=object), size=take, replace=False)
    return pd.DatetimeIndex(sorted(chosen))
