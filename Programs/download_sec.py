#!/usr/bin/env python3
"""
download_edgar_filings.py

Download all 10-K and 10-Q filings for a given company (default: Cal-Maine Foods, Ticker: CALM)
from the SEC EDGAR website, using the official submissions JSON.

Features
- Resolves CIK from ticker via SEC "company_tickers.json"
- Scrapes both "recent" filings and any older chunks listed in submissions files
- Downloads: full submission .txt and the primary document
- Directory layout: ./Data/sec_filings/<TICKER>/
- Filenames: <FORM>_<YYYY-MM-DD>.<ext> (e.g., 10-Q_2025-08-30.htm, 10-Q_2025-08-30.txt)
- Politely rate-limited and sets a descriptive User-Agent as required by the SEC
- Optional inclusion of amendments (10-K/A, 10-Q/A) via --include-amendments

Usage
------
python download_edgar_filings.py --ticker CALM --forms 10-K 10-Q --include-amendments false --outdir Data/sec_filings --email you@example.com --name "Your Name"

Notes
-----
- You MUST provide a descriptive User-Agent (include your email). The --email and --name flags help set that.
"""

import argparse
import os
import time
import re
import json
from typing import Dict, List, Tuple, Iterable
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
try:
    # urllib3 v2+
    from urllib3.util.retry import Retry
    _UB3_USES_ALLOWED = True
except Exception:  # pragma: no cover
    # Fallback for older urllib3
    from urllib3.util import Retry  # type: ignore
    _UB3_USES_ALLOWED = False

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL_FMT = "https://data.sec.gov/submissions/CIK{cik_padded}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

def build_headers(name: str, email: str, purpose: str = "academic research; download 10-K/10-Q"):
    ua = f"{name} ({email}) - {purpose}".strip()
    headers = {
        "User-Agent": ua,
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }
    if email:
        headers["From"] = email
    return headers

def create_session(headers: Dict, max_retries: int = 5, backoff_factor: float = 0.8) -> requests.Session:
    """Create a configured requests Session with retry/backoff and default headers."""
    session = requests.Session()
    session.headers.update(headers)
    if _UB3_USES_ALLOWED:
        retry = Retry(
            total=max_retries,
            read=max_retries,
            connect=max_retries,
            status=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
    else:  # pragma: no cover
        retry = Retry(
            total=max_retries,
            read=max_retries,
            connect=max_retries,
            status=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            method_whitelist=frozenset(["GET"]),
            raise_on_status=False,
        )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def load_ticker_map(session: requests.Session) -> Dict[str, Dict]:
    """Load SEC ticker map (small JSON) -> dict keyed by lowercase ticker."""
    r = session.get(SEC_TICKERS_URL, timeout=30)
    r.raise_for_status()
    # The JSON is an object with numeric keys mapping to dicts: {"0":{"cik_str":..., "ticker":"A","title":"Agilent"} ...}
    data = r.json()
    out = {}
    for _, rec in data.items():
        t = rec.get("ticker", "").lower()
        if t:
            out[t] = rec
    return out

def resolve_cik_from_ticker(ticker: str, session: requests.Session) -> str:
    mp = load_ticker_map(session)
    rec = mp.get(ticker.lower())
    if not rec:
        raise ValueError(f"Ticker '{ticker}' not found in SEC map.")
    cik_str = str(rec["cik_str"]).strip()
    return cik_str

def get_submissions_json(cik: str, session: requests.Session) -> Dict:
    cik_padded = str(cik).zfill(10)
    url = SEC_SUBMISSIONS_URL_FMT.format(cik_padded=cik_padded)
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def iter_filings_from_submissions(submissions: Dict, session: requests.Session) -> Iterable[Dict]:
    """
    Yield filing dicts with keys: form, accessionNumber, filingDate, primaryDocument.
    Combines 'recent' and any older 'files' JSON chunks.
    """
    # Recent
    filings = submissions.get("filings", {})
    recent = filings.get("recent", {})
    forms = recent.get("form", [])
    accs = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    prims = recent.get("primaryDocument", [])
    for i in range(min(len(forms), len(accs), len(dates), len(prims))):
        yield {
            "form": forms[i],
            "accessionNumber": accs[i],
            "filingDate": dates[i],
            "primaryDocument": prims[i],
        }
    # Older chunks (optional)
    # Example structure: submissions['filings']['files'] = [{"name":"CIK########-index.json","filingCount":..., "filingFrom":..., "filingTo":...}, ...]
    for chunk in filings.get("files", []):
        name = chunk.get("name")
        if not name or not name.endswith(".json"):
            continue
        # Older files live under https://data.sec.gov/submissions/<name>
        older_url = urljoin("https://data.sec.gov/submissions/", name)
        yield from _iter_filings_from_chunk(older_url, session)

def _iter_filings_from_chunk(url: str, session: requests.Session) -> Iterable[Dict]:
    # Pull chunk file, which typically has {"filings": [{"form": "...", "accessionNumber": "...", "filingDate": "...", "primaryDocument": "..."}, ...]}
    # But formats vary; we handle common cases.
    # Use caller-provided session to comply with SEC requirements (descriptive User-Agent with contact email).
    r = session.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    filings = data.get("filings", [])
    for f in filings:
        if isinstance(f, dict) and all(k in f for k in ("form", "accessionNumber", "filingDate", "primaryDocument")):
            yield f

def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def readable_basename(form: str, date_str: str, ext: str) -> str:
    """Create a readable base filename: <FORM>_<YYYY-MM-DD>.<ext>.

    If the form has a trailing "/A", include it literally (e.g., 10-K_A_2023-06-03.htm).
    """
    safe_form = sanitize(form).replace("/", "_")
    safe_date = sanitize(date_str)
    if not ext.startswith('.'):
        ext = '.' + ext
    return f"{safe_form}_{safe_date}{ext}"

def download_one_filing(cik: str, ticker: str, filing: Dict, outdir: str, session: requests.Session) -> Tuple[str, List[str]]:
    """
    Download the index page, full submission .txt, and primary document for a single filing.
    Returns (filing_dir, saved_files).
    """
    form = filing["form"]
    accession_dashed = filing["accessionNumber"]
    accession = accession_dashed.replace("-", "")
    date = filing["filingDate"]
    primary = filing.get("primaryDocument", "")
    cik_nolead = str(int(cik))  # strip leading zeros
    base = f"{SEC_ARCHIVES_BASE}/{cik_nolead}/{accession}/"

    ticker_dir = os.path.join(outdir, ticker.upper())
    os.makedirs(ticker_dir, exist_ok=True)

    saved = []

    # 1) Full submission text (readable name)
    txt_url = urljoin(base, f"{accession}.txt")
    txt_filename = readable_basename(form, date, "txt")
    txt_path = os.path.join(ticker_dir, txt_filename)
    _fetch(txt_url, txt_path, session, ok_404=True, label="submission")
    if os.path.exists(txt_path):
        saved.append(txt_path)

    # 2) Primary document (often HTML, readable name)
    if primary:
        prim_url = urljoin(base, primary)
        # choose extension from primary name; default to .htm
        _, dot, ext = sanitize(primary).rpartition('.')
        ext = ext if dot else 'htm'
        prim_filename = readable_basename(form, date, ext)
        prim_path = os.path.join(ticker_dir, prim_filename)
        _fetch(prim_url, prim_path, session, ok_404=True, label="primary")
        if os.path.exists(prim_path):
            saved.append(prim_path)

    return ticker_dir, saved

def _fetch(url: str, path: str, session: requests.Session, ok_404: bool = False, label: str = ""):
    """Fetch a URL to a local file with simple retry/backoff on transient server errors.

    Treat 404 as optional when ok_404 is True. On repeated transient failures (e.g., 429/5xx),
    retry with exponential backoff and ultimately skip if ok_404 is True to avoid halting the run.
    """
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return

    r = session.get(url, timeout=60)
    if r.status_code == 404 and ok_404:
        return
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return

def main():
    p = argparse.ArgumentParser(description="Download all 10-K and 10-Q filings for a given company (default: CALM).")
    p.add_argument("--ticker", default="CALM", help="Ticker symbol, e.g., CALM")
    p.add_argument("--forms", nargs="*", default=["10-K", "10-Q"], help="Form types to include, e.g., 10-K 10-Q")
    p.add_argument("--include-amendments", default="false", choices=["true", "false"], help="Include 10-K/A and 10-Q/A")
    p.add_argument("--outdir", default=os.path.join("Data", "sec_filings"), help="Output directory")
    p.add_argument("--retries", type=int, default=6, help="Max retries for transient errors (HTTPAdapter)")
    p.add_argument("--max-filings", type=int, default=0, help="Optional cap on number of filings to download (0 = no cap)")
    p.add_argument("--name", default="Research Script", help="Your name for SEC User-Agent")
    p.add_argument("--email", default="chenningxuecon@gmail.com", help="Your email for SEC User-Agent")

    args = p.parse_args()

    include_amends = (args.include_amendments.lower() == "true")
    forms_wanted = set([f.upper() for f in args.forms])
    if include_amends:
        forms_wanted |= {f + "/A" for f in list(forms_wanted)}

    headers = build_headers(args.name, args.email)
    session = create_session(headers, max_retries=max(1, args.retries))

    print(f"[i] Resolving CIK for ticker {args.ticker} ...")
    cik = resolve_cik_from_ticker(args.ticker, session)
    print(f"[i] CIK = {cik}")

    print("[i] Fetching submissions JSON ...")
    # quick connectivity probe before heavy loop
    try:
        _ = get_submissions_json(cik, session)
    except Exception as e:
        print(f"[w] Connectivity or access issue fetching submissions: {e}")
        print("[w] Ensure your User-Agent includes your real name and email and try again later.")
        raise
    subs = get_submissions_json(cik, session)

    print("[i] Enumerating filings ...")
    filings = list(iter_filings_from_submissions(subs, session))

    # Filter to wanted forms
    selected = [f for f in filings if f.get("form", "").upper() in forms_wanted]
    # Sort by date ascending
    selected.sort(key=lambda f: f.get("filingDate", ""))
    if args.max_filings and args.max_filings > 0:
        selected = selected[: args.max_filings]

    print(f"[i] Found {len(selected)} filings of types {sorted(forms_wanted)}")

    # Download loop
    total_saved = 0
    for f in selected:
        form = f["form"]
        date = f["filingDate"]
        acc = f["accessionNumber"]
        print(f"[i] Downloading {form} {date} {acc} ...")
        _, saved = download_one_filing(cik, args.ticker, f, args.outdir, session)
        total_saved += len(saved)

    print(f"[✓] Done. Saved/verified files: {total_saved}.")
    print(f"[✓] Output root: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()
