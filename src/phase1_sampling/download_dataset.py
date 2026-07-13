"""
download_dataset.py — CIC IoT-DIAD 2024 targeted downloader
ECE 597 Capstone, Phase 1 (Step 1)

Crawls the CIC browse portal using YOUR logged-in browser session cookie,
lists every file it finds, filters to the files this project needs, and
downloads them into the local data folder.

-----------------------------------------------------------------------------
HOW TO GET YOUR COOKIE (Chrome, ~10 seconds):
  1. Open https://cicresearch.ca/IOTDataset/CIC-IoT-IDAD-Dataset-2024/browse.php?p=
     and make sure the folder listing loads.
  2. Press F12 -> Network tab -> refresh the page -> click the 'browse.php' row.
  3. Under Request Headers, copy the full value of 'Cookie:'.
  4. Paste it into COOKIE below (keep the quotes).

USAGE:
  python download_dataset.py --list              # dry run: show full file tree
  python download_dataset.py --download          # download only project files
  python download_dataset.py --download --all    # download everything (big!)
-----------------------------------------------------------------------------
"""

import argparse
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, unquote

import requests
from bs4 import BeautifulSoup

# ============================= CONFIG =======================================
BASE = "https://cicresearch.ca/IOTDataset/CIC-IoT-IDAD-Dataset-2024/"
START = BASE + "browse.php?p="

COOKIE = "Token=375songupsu2ptuh8s5ac0fk2i"          # <-- from Chrome DevTools (step above)

DATA_ROOT = Path(r"C:\ece597-data\raw")    # local, non-OneDrive storage

# Which files the PROJECT actually needs. A file is kept if its lowercased
# path matches at least one INCLUDE pattern and no EXCLUDE pattern.
INCLUDE_PATTERNS = [
    r"benign",
    r"ddos.*http.*flood",      # DDoS-HTTP Flood
    r"(?<!d)dos.*http.*flood", # DoS-HTTP Flood (not preceded by another 'd')
    r"dns.*spoof",
    r"xss",
    r"brute.*force",
    r"readme",                 # keep the README.txt feature descriptions
]
EXCLUDE_PATTERNS = [
    r"tcp.*flood",             # TCP Flood variants: no flow data, not used
    r"udp",                    # instructor: TCP files, not UDP
]
FILE_EXTENSIONS = (".csv", ".zip", ".txt", ".rar", ".7z")
# ============================================================================

session = requests.Session()
session.headers.update({
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"),
    "Cookie": COOKIE,
})
# Ask the server to close the connection after each response instead of
# keeping it alive; CIC's server kills long-lived connections, which is
# what caused the RemoteDisconnected error.
session.headers["Connection"] = "close"


def crawl(url, seen=None, depth=0):
    """Recursively walk browse.php folder links; yield (file_url, rel_path)."""
    if seen is None:
        seen = set()
    if url in seen or depth > 6:
        return
    seen.add(url)

    r = session.get(url, timeout=60)
    r.raise_for_status()
    if "DATASET DOWNLOAD FORM" in r.text:
        sys.exit("ERROR: portal returned the registration form -> your cookie "
                 "is missing/expired. Re-copy it from DevTools and try again.")

    soup = BeautifulSoup(r.text, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(url, href)
        if not full.startswith(BASE):
            continue                          # off-site link (logos, UNB, etc.)
        low = unquote(full).lower()
        if low.endswith(FILE_EXTENSIONS):
            rel = unquote(full[len(BASE):].split("browse.php")[-1].lstrip("?p="))
            yield full, rel
        elif "browse.php?p=" in full and full not in seen:
            time.sleep(0.3)                   # be polite to their server
            yield from crawl(full, seen, depth + 1)


def wanted(rel_path):
    low = rel_path.lower()
    if any(re.search(p, low) for p in EXCLUDE_PATTERNS):
        return False
    return any(re.search(p, low) for p in INCLUDE_PATTERNS)


def local_path(rel_path):
    """Route packet-based files to raw/packet, flow-based to raw/flow."""
    rel = unquote(rel_path).replace("+", " ")
    rel = rel.split("download.php?file=")[-1]      # strip the PHP prefix
    parts = [p for p in rel.replace("\\", "/").split("/") if p]
    low = parts[0].lower()
    sub = "flow" if "flow based" in low else "packet"
    fname = parts[-1]
    category = parts[-2] if len(parts) > 2 else "root"   # READMEs -> root/
    # sanitize anything Windows can't have in a folder name
    category = re.sub(r'[<>:"/\\|?*]', "_", category)
    return DATA_ROOT / sub / category / fname


def download(url, dest: Path, max_retries=5):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (exists): {dest.name}")
        return

    tmp = dest.with_suffix(dest.suffix + ".part")   # partial file guard
    for attempt in range(1, max_retries + 1):
        try:
            with session.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                done = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r  {dest.name}: {done/total:6.1%}", end="")
            tmp.rename(dest)                        # only now is it "real"
            print(f"\r  {dest.name}: done ({done/1e6:.1f} MB)      ")
            time.sleep(1.0)                         # breathe between files
            return
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.Timeout) as e:
            wait = 5 * attempt                      # 5s, 10s, 15s ... backoff
            print(f"\n  connection dropped ({type(e).__name__}), "
                  f"retry {attempt}/{max_retries} in {wait}s ...")
            tmp.unlink(missing_ok=True)             # discard partial data
            time.sleep(wait)

    raise RuntimeError(f"Failed after {max_retries} retries: {dest.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (exists): {dest.name}")
        return
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):   # 1 MB
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {dest.name}: {done/total:6.1%}", end="")
        print(f"\r  {dest.name}: done ({done/1e6:.1f} MB)      ")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="dry run: print file tree")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--all", action="store_true", help="ignore filters, take everything")
    args = ap.parse_args()

    if COOKIE.startswith("PASTE"):
        sys.exit("Edit COOKIE at the top of this file first (see header).")

    print("Crawling portal ... (this takes a minute)\n")
    files = list(crawl(START))
    print(f"Found {len(files)} files total.\n")

    for url, rel in files:
        mark = "KEEP" if (args.all or wanted(rel)) else "----"
        print(f" [{mark}] {unquote(rel)}")

    if args.download:
        targets = [(u, r) for u, r in files if args.all or wanted(r)]
        print(f"\nDownloading {len(targets)} files to {DATA_ROOT} ...\n")
        for url, rel in targets:
            download(url, local_path(rel))
        print("\nAll done. Verify sizes, then you're ready for generate_dataset.py.")
