"""
Simplified GPT-only extractor for National pasture-raised shell egg items.

What it does:
  - Reads text from PDFs in a directory (first N pages per file)
  - Sends the combined text to an OpenAI chat model
  - Asks for JSON rows: date (YYYY-MM-DD), item, quantity (6/12/18), cw_wtd_avg
  - Keeps only the latest week and writes CSV

Run:
  python Programs/extract_pasture_raised_simple.py \
    --pdf-dir Data/usda/retail_reports \
    --output Data/pasture_raised_latest.csv \
    --model gpt-5

Requires:
  - OPENAI_API_KEY in environment or .env (repo root or Programs/.env)
  - pip install pdfplumber pandas
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd

try:
    import pdfplumber  # type: ignore
except Exception as exc:
    raise SystemExit("Install pdfplumber: pip install pdfplumber") from exc

import urllib.request
from urllib.error import HTTPError, URLError
import re


PROMPT = (
    "You are given raw text from ONE USDA weekly retail egg PDF.\n"
    "Extract ONLY National Shell Egg pasture raised items from THIS report.\n"
    "Use ONLY the current week price (CW Wtd Avg). Do NOT use prior week (PW), last week, or any non-CW value.\n"
    "Return strict JSON: {\"rows\": [{\"date\": \"YYYY-MM-DD\", \"item\": string, \"quantity\": 6|12|18, \"cw_wtd_avg\": number}, ...]}\n"
    "Rules:\n"
    "- National section only (ignore regional data).\n"
    "- Item: concise name (no 'Fresh', store counts, or price ranges).\n"
    "- Pasture-raised must be explicit. Include an item only if its description contains 'pasture-raised', 'pasture raised', or 'pastured'.\n"
    "  Do NOT include items that are only 'Organic', 'Cage-Free', 'Free Range', 'Large Brown', etc., unless they explicitly say pasture-raised/pastured.\n"
    "- Quantity: number of eggs (6, 12, or 18).\n"
    "- cw_wtd_avg: numeric case-weighted average for CW only.\n"
    "- If none found, return {\"rows\": []}."
)


def load_env_file(env_path: Path) -> None:
    try:
        if not env_path.exists() or not env_path.is_file():
            return
        for line in env_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.lower().startswith("export "):
                s = s[7:].lstrip()
            if "=" not in s:
                continue
            key, val = s.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        pass


def load_env_defaults() -> None:
    here = Path(__file__).resolve()
    repo_root = here.parents[1] if len(here.parents) >= 2 else Path.cwd()
    for p in (repo_root / ".env", repo_root / "Programs" / ".env", Path.cwd() / ".env"):
        load_env_file(p)


def read_pdf_text(pdf_path: Path, max_pages: int) -> str:
    chunks: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages[: max(1, max_pages)]
        for page in pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t:
                chunks.append(t)
    return "\n\n".join(chunks)


def parse_week_ending(text: str) -> Optional[pd.Timestamp]:
    # Try common patterns seen in USDA reports and pick the later date if a range is given
    patterns = (
        r"week\s+(?:ending|ended|end)\s*[:\-]?\s*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})",
        r"for\s+week\s+(?:ending|ended)\s*[:\-]?\s*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})",
        r"week\s+of\s*[:\-]?\s*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})\s*(?:to|\-)\s*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})",
        r"(\d{1,2}/\d{1,2}/\d{2,4})\s*(?:to|\-)\s*(\d{1,2}/\d{1,2}/\d{2,4})",
    )
    compact = " ".join((text or "").split())
    for pat in patterns:
        m = __import__('re').search(pat, compact, flags=__import__('re').I)
        if not m:
            continue
        groups = [g for g in m.groups() if g]
        dates = pd.to_datetime(groups, errors="coerce")
        dates = [d for d in dates if pd.notna(d)]
        if dates:
            return pd.to_datetime(max(dates)).normalize()
    # fallback: pick latest Month DD, YYYY anywhere
    candidates = __import__('re').findall(r"([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})", compact)
    if candidates:
        dates = pd.to_datetime(candidates, errors="coerce")
        dates = [d for d in dates if pd.notna(d)]
        if dates:
            return pd.to_datetime(max(dates)).normalize()
    return None


def call_openai(api_key: str, model: str, text: str) -> Optional[List[dict]]:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    messages = [
        {"role": "system", "content": "You extract structured data from USDA PDF text."},
        {"role": "user", "content": PROMPT + "\n<PDF_TEXT>\n" + text[:180000] + "\n</PDF_TEXT>"},
    ]
    payload = {"model": model, "messages": messages, "temperature": 1, "response_format": {"type": "json_object"}}

    def _send(use_rf: bool) -> Optional[str]:
        body = dict(payload)
        if not use_rf:
            body.pop("response_format", None)
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8")
        except HTTPError as e:
            try:
                err = e.read().decode("utf-8", errors="ignore")
            except Exception:
                err = ""
            err_snippet = err[:400].replace("\n", " ")
            print(f"[WARN] OpenAI HTTP {e.code}: {e.reason} | body={err_snippet}")
            return None
        except URLError as e:
            print(f"[WARN] OpenAI URL error: {e.reason}")
            return None
        except Exception as exc:
            print(f"[WARN] OpenAI error: {exc}")
            return None

    raw = _send(True) or _send(False)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
        obj = json.loads(content)
        rows = obj.get("rows", [])
        return rows if isinstance(rows, list) else []
    except Exception as exc:
        print(f"[WARN] Failed to parse JSON content: {exc}")
        return None


PASTURE_EXPLICIT_RE = re.compile(r"\bpasture[\s-]?raised\b|\bpastured\b", re.I)


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_env_defaults()

    parser = argparse.ArgumentParser(description="GPT-only pasture-raised extractor from PDFs")
    parser.add_argument("--pdf-dir", type=str, default=str(Path("Data/usda/retail_reports")), help="Directory of USDA PDFs")
    parser.add_argument("--output", type=str, default=str(Path("Data/pasture_raised_panel.csv")), help="Output CSV path")
    parser.add_argument("--model", type=str, default="gpt-5", help="OpenAI model")
    parser.add_argument("--max-pages", type=int, default=3, help="Pages per PDF to read (from start)")
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel workers")
    args = parser.parse_args(argv)

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set. Put it in .env or environment.")

    pdf_dir = Path(args.pdf_dir)
    output_csv = Path(args.output)
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        raise SystemExit(f"PDF directory not found: {pdf_dir}")

    # Incremental append: after each PDF, append CW-based pasture rows from that PDF
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    existing_df: Optional[pd.DataFrame] = None
    if output_csv.exists():
        try:
            existing_df = pd.read_csv(output_csv)
            if "date" in existing_df.columns:
                existing_df["date"] = pd.to_datetime(existing_df["date"], errors="coerce").dt.date.astype(str)
            if "quantity" in existing_df.columns:
                existing_df["quantity"] = pd.to_numeric(existing_df["quantity"], errors="coerce").astype("Int64")
            if "cw_wtd_avg" in existing_df.columns:
                existing_df["cw_wtd_avg"] = pd.to_numeric(existing_df["cw_wtd_avg"], errors="coerce")
            if "item" in existing_df.columns:
                existing_df = existing_df[existing_df["item"].astype(str).str.contains(PASTURE_EXPLICIT_RE, na=False)]
        except Exception:
            existing_df = None

    pdf_list = sorted(pdf_dir.glob("*.pdf"))

    def process_one(pdf_path: Path) -> Optional[pd.DataFrame]:
        print(f"[INFO] Processing {pdf_path.name}")
        text_local = read_pdf_text(pdf_path, max_pages=args.max_pages)
        rows_local = call_openai(api_key=api_key, model=args.model, text=text_local) or []
        if not rows_local:
            return None
        df_local = pd.DataFrame(rows_local, columns=["date", "item", "quantity", "cw_wtd_avg"])  # enforce columns
        if df_local.empty:
            return None
        pdf_week_local = parse_week_ending(text_local)
        dates = pd.to_datetime(df_local["date"], errors="coerce")
        if pd.notna(pdf_week_local):
            dates = dates.fillna(pdf_week_local)
        df_local["date"] = dates.dt.date.astype(str)
        df_local["item"] = df_local["item"].astype(str).str.strip()
        df_local["quantity"] = pd.to_numeric(df_local["quantity"], errors="coerce").astype("Int64")
        df_local["cw_wtd_avg"] = pd.to_numeric(df_local["cw_wtd_avg"], errors="coerce")
        df_local = df_local[df_local["item"].str.contains(PASTURE_EXPLICIT_RE, na=False)]
        return df_local[["date", "item", "quantity", "cw_wtd_avg"]]

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(process_one, p): p for p in pdf_list}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                new_df = fut.result()
            except Exception as exc:
                print(f"[WARN] Failed to process {p.name}: {exc}")
                continue
            if new_df is None or new_df.empty:
                continue
            with lock:
                parts: List[pd.DataFrame] = []
                if existing_df is not None and not existing_df.empty:
                    parts.append(existing_df[["date", "item", "quantity", "cw_wtd_avg"]])
                parts.append(new_df)
                combined = pd.concat(parts, ignore_index=True)
                combined = combined.drop_duplicates(subset=["date", "item", "quantity", "cw_wtd_avg"]).reset_index(drop=True)
                combined.to_csv(output_csv, index=False)
                existing_df = combined
                print(f"[INFO] Appended rows from {p.name}; file now has {len(combined)} rows")

    # Ensure file exists even if no rows were found
    if not output_csv.exists():
        pd.DataFrame(columns=["date", "item", "quantity", "cw_wtd_avg"]).to_csv(output_csv, index=False)
        print(f"[INFO] Wrote 0 rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


