"""PDF ingestion utilities for AI Stock Ranks.

The ranking pipeline expects a TickerRow-style table. This module builds that
table from per-ticker PDF documents (financial statements and investor decks).
"""

from __future__ import annotations

import glob
import html
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from pypdf import PdfReader  # type: ignore
except ImportError:  # pragma: no cover
    PdfReader = None

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover
    OpenAI = None

try:
    import openai as openai_legacy  # type: ignore
except ImportError:  # pragma: no cover
    openai_legacy = None


TICKER_COLUMNS = [
    "ticker",
    "sector",
    "market_cap",
    "pe",
    "ev_ebitda",
    "debt_equity",
    "rev_growth",
    "fcf_margin",
    "price_momentum",
    "volatility",
    "timestamp",
    "source",
]

NUMERIC_COLUMNS = [
    "market_cap",
    "pe",
    "ev_ebitda",
    "debt_equity",
    "rev_growth",
    "fcf_margin",
    "price_momentum",
    "volatility",
]


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_prompt_path() -> str:
    return os.path.join(
        _project_root(), "prompts", "extract_ticker_metrics_from_pdf_v1.md"
    )


def _load_prompt(prompt_path: Optional[str]) -> str:
    path = prompt_path or _default_prompt_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PDF extraction prompt not found at: {path}. "
            "Provide --pdf-prompt or create the default prompt file."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _extract_json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("Could not find JSON object in model response.")
    return json.loads(text[start : end + 1])


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _to_float(value: Any) -> float:
    if value is None:
        return float(np.nan)
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip()
    if raw == "":
        return float(np.nan)
    cleaned = (
        raw.replace(",", "")
        .replace("$", "")
        .replace("%", "")
        .replace("x", "")
        .replace("X", "")
    )
    try:
        return float(cleaned)
    except ValueError:
        return float(np.nan)


def _extract_text_from_pdf(path: str, max_pages: int) -> str:
    if PdfReader is None:
        raise ImportError(
            "pypdf is required for PDF ingestion. Add `pypdf` to dependencies."
        )
    reader = PdfReader(path)
    pages: List[str] = []
    for i, page in enumerate(reader.pages):
        if i >= max_pages:
            break
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = _normalize_space(text)
        if text:
            pages.append(text)
    return "\n".join(pages)


def _extract_text_from_html(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    without_script = re.sub(
        r"<(script|style)\b[^>]*>.*?</\1>", " ", raw, flags=re.IGNORECASE | re.DOTALL
    )
    without_tags = re.sub(r"<[^>]+>", " ", without_script)
    return _normalize_space(html.unescape(without_tags))


def _extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return _normalize_space(f.read())


def _extract_text_from_document(path: str, max_pages: int) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _extract_text_from_pdf(path, max_pages=max_pages)
    if ext in {".htm", ".html", ".xml"}:
        return _extract_text_from_html(path)
    if ext in {".txt", ".md"}:
        return _extract_text_from_txt(path)
    return ""


def _infer_ticker_from_filename(filename: str) -> Optional[str]:
    match = re.match(r"([A-Za-z][A-Za-z0-9.\-]{0,9})[_\-\s]", filename)
    if not match:
        return None
    return match.group(1).upper()


def _infer_ticker_from_path_root_segment(segment: str) -> Optional[str]:
    match = re.match(r"([A-Za-z][A-Za-z0-9.\-]{0,9})", segment)
    if not match:
        return None
    return match.group(1).upper()


def _discover_ticker_pdf_groups(pdf_root: str) -> Dict[str, List[str]]:
    patterns = ["*.pdf", "*.htm", "*.html", "*.xml", "*.txt"]
    doc_paths: List[str] = []
    for pattern in patterns:
        doc_paths.extend(
            glob.glob(os.path.join(pdf_root, "**", pattern), recursive=True)
        )
    doc_paths = sorted(set(doc_paths))
    if not doc_paths:
        raise ValueError(
            f"No supported documents found under: {pdf_root} "
            "(supported: pdf, htm, html, xml, txt)"
        )

    grouped: Dict[str, List[str]] = {}
    for doc_path in doc_paths:
        rel = os.path.relpath(doc_path, pdf_root)
        parts = rel.split(os.sep)
        ticker: Optional[str] = None
        if len(parts) > 1:
            ticker = _infer_ticker_from_path_root_segment(parts[0])
        if not ticker:
            ticker = _infer_ticker_from_filename(os.path.basename(doc_path))
        if not ticker:
            raise ValueError(
                "Could not infer ticker for document path. Use either a ticker folder name "
                "or file prefix like `AAPL_q4_2025.pdf`: "
                f"{doc_path}"
            )
        grouped.setdefault(ticker, []).append(doc_path)
    return grouped


def _build_user_payload(ticker: str, source_docs: List[str], text: str) -> str:
    docs = "\n".join(f"- {name}" for name in source_docs)
    return (
        f"ticker: {ticker}\n"
        f"documents:\n{docs}\n\n"
        "Extracted document text:\n"
        f"{text}"
    )


def _call_extractor_model(
    system_prompt: str,
    ticker: str,
    source_docs: List[str],
    extracted_text: str,
    model: str,
    temperature: float,
) -> Optional[Dict[str, Any]]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": _build_user_payload(ticker, source_docs, extracted_text),
        },
    ]

    if OpenAI is not None:
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            content = response.choices[0].message.content or ""
            return _extract_json_object(content)
        except Exception:
            pass

    if openai_legacy is not None:
        try:
            openai_legacy.api_key = api_key
            response = openai_legacy.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=900,
            )
            content = response.choices[0].message["content"]  # type: ignore[index]
            return _extract_json_object(content)
        except Exception:
            pass

    return None


def _coerce_ticker_row(
    ticker: str,
    extracted: Optional[Dict[str, Any]],
    source_docs: List[str],
    asof_timestamp: str,
) -> Dict[str, Any]:
    payload = extracted or {}
    row: Dict[str, Any] = {
        "ticker": ticker,
        "sector": str(payload.get("sector", "") or "").strip(),
        "timestamp": str(payload.get("timestamp", "") or asof_timestamp).strip(),
        "source": "; ".join(source_docs),
    }
    for col in NUMERIC_COLUMNS:
        row[col] = _to_float(payload.get(col))
    if not row["timestamp"]:
        row["timestamp"] = asof_timestamp
    return row


def build_ticker_dataframe_from_pdf_root(
    pdf_root: str,
    prompt_path: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_pages_per_pdf: int = 40,
    max_chars_per_ticker: int = 50000,
    asof_timestamp: Optional[str] = None,
) -> pd.DataFrame:
    """Build a TickerRow DataFrame from documents.

    Expected directory pattern (recommended):
    - <pdf_root>/<TICKER>/*.(pdf|htm|html|xml|txt)

    Also supported:
    - <pdf_root>/AAPL_q4_2025.pdf (ticker inferred from filename prefix)
    """
    if not os.path.isdir(pdf_root):
        raise ValueError(f"PDF root must be a directory: {pdf_root}")

    system_prompt = _load_prompt(prompt_path)
    grouped = _discover_ticker_pdf_groups(pdf_root)
    effective_asof = asof_timestamp or _iso_utc_now()
    rows: List[Dict[str, Any]] = []

    for ticker in sorted(grouped.keys()):
        docs = grouped[ticker]
        source_doc_names = [os.path.basename(p) for p in docs]
        text_parts: List[str] = []
        consumed = 0
        for path in docs:
            piece = _extract_text_from_document(path, max_pages=max_pages_per_pdf)
            if not piece:
                continue
            labeled = f"\n\n### FILE: {os.path.basename(path)}\n{piece}"
            remaining = max_chars_per_ticker - consumed
            if remaining <= 0:
                break
            if len(labeled) > remaining:
                labeled = labeled[:remaining]
            text_parts.append(labeled)
            consumed += len(labeled)
        combined_text = "".join(text_parts).strip()
        extracted = _call_extractor_model(
            system_prompt=system_prompt,
            ticker=ticker,
            source_docs=source_doc_names,
            extracted_text=combined_text,
            model=model,
            temperature=temperature,
        )
        row = _coerce_ticker_row(
            ticker=ticker,
            extracted=extracted,
            source_docs=source_doc_names,
            asof_timestamp=effective_asof,
        )
        rows.append(row)

    if not rows:
        raise ValueError(f"No ticker rows could be built from: {pdf_root}")

    df = pd.DataFrame(rows)
    for col in TICKER_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[TICKER_COLUMNS].copy()
