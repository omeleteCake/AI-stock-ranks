"""Download recent SEC filing documents for one or more tickers.

This script prepares document folders for the ingestion pipeline:
`pdf_inputs/<TICKER>/*.(pdf|htm|html|xml|txt)`

Typical usage:

    python -m src.fetch_latest_sec_pdf --tickers AAPL,MSFT --out-dir pdf_inputs
    python -m src.fetch_latest_sec_pdf --tickers-csv tickers.csv --out-dir pdf_inputs

After downloading, run:

    python -m src.orchestrator pdf_inputs --input-mode pdf
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import quote

import pandas as pd
import requests


DEFAULT_FORMS = ("10-Q", "10-K", "20-F", "6-K", "8-K")
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVE_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/index.json"
SEC_ARCHIVE_FILE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{name}"


@dataclass
class FilingRef:
    cik: str
    accession_nodash: str
    accession_display: str
    form: str
    filing_date: str
    primary_document: str


@dataclass
class DownloadRecord:
    ticker: str
    cik: str
    form: str
    filing_date: str
    accession: str
    file_name: str
    sec_url: str
    local_path: str
    status: str
    note: str = ""


def _clean_ticker(raw: str) -> str:
    return raw.strip().upper().replace(".", "-")


def _safe_filename(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw.strip())
    return cleaned[:180] if cleaned else "document.pdf"


def _session(user_agent: str) -> requests.Session:
    sess = requests.Session()
    sess.headers.update(
        {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json,text/html,application/xml,*/*",
        }
    )
    return sess


def _load_tickers(args: argparse.Namespace) -> List[str]:
    tickers: List[str] = []
    if args.tickers:
        tickers.extend(_clean_ticker(x) for x in args.tickers.split(",") if x.strip())
    if args.tickers_csv:
        df = pd.read_csv(args.tickers_csv)
        if "ticker" not in df.columns:
            raise ValueError("Ticker CSV must include a `ticker` column.")
        tickers.extend(_clean_ticker(x) for x in df["ticker"].astype(str).tolist())
    unique = sorted(set(t for t in tickers if t))
    if not unique:
        raise ValueError("No tickers supplied. Use --tickers or --tickers-csv.")
    return unique


def _load_ticker_cik_map(sess: requests.Session) -> Dict[str, str]:
    resp = sess.get(SEC_TICKERS_URL, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    mapping: Dict[str, str] = {}

    if isinstance(payload, dict):
        for value in payload.values():
            if not isinstance(value, dict):
                continue
            ticker = str(value.get("ticker", "")).upper()
            cik_num = value.get("cik_str")
            if not ticker or cik_num is None:
                continue
            cik = str(cik_num).zfill(10)
            mapping[ticker] = cik
    return mapping


def _recent_filings_for_cik(
    sess: requests.Session,
    cik10: str,
    forms: Sequence[str],
    filings_per_ticker: int,
) -> List[FilingRef]:
    forms_set = {f.upper() for f in forms}
    url = SEC_SUBMISSIONS_URL.format(cik=cik10)
    resp = sess.get(url, timeout=30)
    if resp.status_code != 200:
        return []
    data = resp.json()
    recent = data.get("filings", {}).get("recent", {})
    form_list = recent.get("form", [])
    accession_list = recent.get("accessionNumber", [])
    filing_date_list = recent.get("filingDate", [])
    primary_doc_list = recent.get("primaryDocument", [])

    found: List[FilingRef] = []
    for i, form in enumerate(form_list):
        form_upper = str(form).upper()
        if form_upper not in forms_set:
            continue
        accession_display = str(accession_list[i])
        accession_nodash = accession_display.replace("-", "")
        filing_date = str(filing_date_list[i])
        primary_document = str(primary_doc_list[i]) if i < len(primary_doc_list) else ""
        found.append(
            FilingRef(
                cik=str(int(cik10)),
                accession_nodash=accession_nodash,
                accession_display=accession_display,
                form=form_upper,
                filing_date=filing_date,
                primary_document=primary_document,
            )
        )
        if len(found) >= filings_per_ticker:
            break
    return found


def _filing_pdf_files(
    sess: requests.Session,
    filing: FilingRef,
) -> List[str]:
    url = SEC_ARCHIVE_INDEX_URL.format(cik=filing.cik, accession=filing.accession_nodash)
    resp = sess.get(url, timeout=30)
    if resp.status_code != 200:
        return []
    payload = resp.json()
    items = payload.get("directory", {}).get("item", [])
    names = [
        str(item.get("name", ""))
        for item in items
        if str(item.get("name", "")).lower().endswith(".pdf")
    ]
    if not names:
        return []

    def score(name: str) -> int:
        low = name.lower()
        s = 0
        if filing.form.lower().replace("-", "") in low:
            s += 10
        if "financial" in low or "annual" in low or "quarter" in low:
            s += 6
        if "presentation" in low or "investor" in low or "deck" in low:
            s += 5
        if "ex99" in low or "ex-99" in low:
            s += 4
        if "press" in low:
            s -= 1
        return s

    ranked = sorted(names, key=lambda n: (score(n), len(n)), reverse=True)
    return ranked


def _download_primary_document_fallback(
    sess: requests.Session,
    filing: FilingRef,
    ticker: str,
    ticker_dir: str,
    skip_existing: bool,
) -> Optional[DownloadRecord]:
    primary = filing.primary_document.strip()
    if not primary:
        return None
    ext = os.path.splitext(primary)[1].lower()
    if ext not in {".htm", ".html", ".xml", ".txt"}:
        return None

    encoded_name = quote(primary)
    sec_url = SEC_ARCHIVE_FILE_URL.format(
        cik=filing.cik,
        accession=filing.accession_nodash,
        name=encoded_name,
    )
    local_name = _safe_filename(f"{filing.filing_date}_{filing.form}_{primary}")
    local_path = os.path.join(ticker_dir, local_name)
    if skip_existing and os.path.exists(local_path):
        return DownloadRecord(
            ticker=ticker,
            cik=filing.cik,
            form=filing.form,
            filing_date=filing.filing_date,
            accession=filing.accession_display,
            file_name=primary,
            sec_url=sec_url,
            local_path=local_path,
            status="skipped",
            note="already_exists_primary_document",
        )
    ok = _download_file(sess, sec_url, local_path)
    return DownloadRecord(
        ticker=ticker,
        cik=filing.cik,
        form=filing.form,
        filing_date=filing.filing_date,
        accession=filing.accession_display,
        file_name=primary,
        sec_url=sec_url,
        local_path=local_path,
        status="downloaded" if ok else "failed",
        note="primary_document_fallback" if ok else "primary_document_download_failed",
    )


def _download_file(sess: requests.Session, url: str, out_path: str) -> bool:
    resp = sess.get(url, timeout=60, stream=True)
    if resp.status_code != 200:
        return False
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 64):
            if chunk:
                f.write(chunk)
    return True


def _record_to_row(r: DownloadRecord) -> Dict[str, Any]:
    return {
        "ticker": r.ticker,
        "cik": r.cik,
        "form": r.form,
        "filing_date": r.filing_date,
        "accession": r.accession,
        "file_name": r.file_name,
        "sec_url": r.sec_url,
        "local_path": r.local_path,
        "status": r.status,
        "note": r.note,
    }


def _write_manifest(path: str, rows: List[DownloadRecord]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "ticker",
        "cik",
        "form",
        "filing_date",
        "accession",
        "file_name",
        "sec_url",
        "local_path",
        "status",
        "note",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_record_to_row(row))


def download_sec_pdfs(
    tickers: Iterable[str],
    out_dir: str,
    user_agent: str,
    forms: Sequence[str] = DEFAULT_FORMS,
    filings_per_ticker: int = 2,
    pdfs_per_filing: int = 2,
    delay_seconds: float = 0.25,
    skip_existing: bool = True,
) -> List[DownloadRecord]:
    sess = _session(user_agent=user_agent)
    ticker_to_cik = _load_ticker_cik_map(sess)
    records: List[DownloadRecord] = []

    for ticker in tickers:
        cik10 = ticker_to_cik.get(ticker)
        if not cik10:
            records.append(
                DownloadRecord(
                    ticker=ticker,
                    cik="",
                    form="",
                    filing_date="",
                    accession="",
                    file_name="",
                    sec_url="",
                    local_path="",
                    status="skipped",
                    note="ticker_not_in_sec_map",
                )
            )
            continue

        filings = _recent_filings_for_cik(
            sess=sess,
            cik10=cik10,
            forms=forms,
            filings_per_ticker=filings_per_ticker,
        )
        if not filings:
            records.append(
                DownloadRecord(
                    ticker=ticker,
                    cik=str(int(cik10)),
                    form="",
                    filing_date="",
                    accession="",
                    file_name="",
                    sec_url="",
                    local_path="",
                    status="skipped",
                    note="no_recent_target_filings",
                )
            )
            continue

        ticker_dir = os.path.join(out_dir, ticker)
        os.makedirs(ticker_dir, exist_ok=True)

        got_any = False
        for filing in filings:
            pdf_files = _filing_pdf_files(sess, filing)
            if not pdf_files:
                fallback_record = _download_primary_document_fallback(
                    sess=sess,
                    filing=filing,
                    ticker=ticker,
                    ticker_dir=ticker_dir,
                    skip_existing=skip_existing,
                )
                if fallback_record is not None:
                    records.append(fallback_record)
                    if fallback_record.status == "downloaded":
                        got_any = True
                else:
                    records.append(
                        DownloadRecord(
                            ticker=ticker,
                            cik=filing.cik,
                            form=filing.form,
                            filing_date=filing.filing_date,
                            accession=filing.accession_display,
                            file_name="",
                            sec_url="",
                            local_path="",
                            status="skipped",
                            note="no_pdf_files_in_filing_directory",
                        )
                    )
                time.sleep(delay_seconds)
                continue

            for file_name in pdf_files[:pdfs_per_filing]:
                encoded_name = quote(file_name)
                sec_url = SEC_ARCHIVE_FILE_URL.format(
                    cik=filing.cik,
                    accession=filing.accession_nodash,
                    name=encoded_name,
                )
                local_name = _safe_filename(
                    f"{filing.filing_date}_{filing.form}_{file_name}"
                )
                local_path = os.path.join(ticker_dir, local_name)

                if skip_existing and os.path.exists(local_path):
                    records.append(
                        DownloadRecord(
                            ticker=ticker,
                            cik=filing.cik,
                            form=filing.form,
                            filing_date=filing.filing_date,
                            accession=filing.accession_display,
                            file_name=file_name,
                            sec_url=sec_url,
                            local_path=local_path,
                            status="skipped",
                            note="already_exists",
                        )
                    )
                    got_any = True
                    continue

                ok = _download_file(sess, sec_url, local_path)
                status = "downloaded" if ok else "failed"
                note = "" if ok else "http_error_or_forbidden"
                records.append(
                    DownloadRecord(
                        ticker=ticker,
                        cik=filing.cik,
                        form=filing.form,
                        filing_date=filing.filing_date,
                        accession=filing.accession_display,
                        file_name=file_name,
                        sec_url=sec_url,
                        local_path=local_path,
                        status=status,
                        note=note,
                    )
                )
                got_any = got_any or ok
                time.sleep(delay_seconds)

        if not got_any:
            records.append(
                DownloadRecord(
                    ticker=ticker,
                    cik=str(int(cik10)),
                    form="",
                    filing_date="",
                    accession="",
                    file_name="",
                    sec_url="",
                    local_path="",
                    status="skipped",
                    note="ticker_has_no_downloaded_documents",
                )
            )
    return records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download recent SEC filing documents into per-ticker folders for "
            "the AI stock ranker ingestion pipeline."
        )
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated tickers (example: AAPL,MSFT,NVDA).",
    )
    parser.add_argument(
        "--tickers-csv",
        type=str,
        default=None,
        help="CSV path with a `ticker` column.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="pdf_inputs",
        help="Output root directory. Files are saved under <out-dir>/<TICKER>/",
    )
    parser.add_argument(
        "--forms",
        type=str,
        default=",".join(DEFAULT_FORMS),
        help="Comma-separated SEC form types in priority order.",
    )
    parser.add_argument(
        "--filings-per-ticker",
        type=int,
        default=2,
        help="How many recent filings to inspect per ticker.",
    )
    parser.add_argument(
        "--pdfs-per-filing",
        type=int,
        default=2,
        help="How many PDF files to download from each filing directory.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Delay in seconds between SEC requests.",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default=os.environ.get(
            "SEC_USER_AGENT", "AIStockRanks/1.0 (contact: you@example.com)"
        ),
        help="SEC-compliant User-Agent with contact information.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional path for CSV manifest output. Default: <out-dir>/manifest.csv",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Redownload files even if a matching local file already exists.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    tickers = _load_tickers(args)
    forms = [f.strip().upper() for f in args.forms.split(",") if f.strip()]
    if not forms:
        raise ValueError("No forms selected. Provide at least one SEC form type.")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = (
        os.path.abspath(args.manifest)
        if args.manifest
        else os.path.join(out_dir, "manifest.csv")
    )

    print(f"Tickers: {len(tickers)}")
    print(f"Forms: {forms}")
    print(f"Output: {out_dir}")
    print("Downloading...")

    records = download_sec_pdfs(
        tickers=tickers,
        out_dir=out_dir,
        user_agent=args.user_agent,
        forms=forms,
        filings_per_ticker=args.filings_per_ticker,
        pdfs_per_filing=args.pdfs_per_filing,
        delay_seconds=args.delay,
        skip_existing=not args.no_skip_existing,
    )

    _write_manifest(manifest_path, records)
    downloaded = sum(1 for r in records if r.status == "downloaded")
    failed = sum(1 for r in records if r.status == "failed")
    skipped = sum(1 for r in records if r.status == "skipped")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: downloaded={downloaded} failed={failed} skipped={skipped}")
    print(
        "Next step: python -m src.orchestrator "
        f"{out_dir} --input-mode pdf"
    )


if __name__ == "__main__":
    main()
