#!/usr/bin/env python3
"""
Download the 13 practice datasets for the NAS Unseen-Data Competition.

Datasets are hosted on:
  - Newcastle University Data Repository (figshare): 11 datasets
  - University of Edinburgh DataShare:               2 datasets (Cryptic, Windspeed)

After running, datasets land in:
  datasets/
    AddNIST/   train_x.npy  train_y.npy  valid_x.npy  valid_y.npy
               test_x.npy   test_y.npy   metadata
    CIFARTile/ ...
    ...

Usage:
  python download_datasets.py                          # all 13
  python download_datasets.py AddNIST CIFARTile        # selected
  python download_datasets.py --list                   # show registry
  python download_datasets.py --out path/to/datasets   # custom output dir
"""

import argparse
import io
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# Newcastle (figshare) datasets: downloaded as a ZIP via ndownloader.
# Edinburgh DataShare datasets: follow DOI redirect to find the download.
# ---------------------------------------------------------------------------

DATASETS = {
    "AddNIST":     {"host": "ncl", "ncl_id": 24574354, "version": 1,
                    "doi": "https://doi.org/10.25405/data.ncl.24574354.v1"},
    "Language":    {"host": "ncl", "ncl_id": 24574729, "version": 1,
                    "doi": "https://doi.org/10.25405/data.ncl.24574729.v1"},
    "MultNIST":    {"host": "ncl", "ncl_id": 24574678, "version": 1,
                    "doi": "https://doi.org/10.25405/data.ncl.24574678.v1"},
    "CIFARTile":   {"host": "ncl", "ncl_id": 24551539, "version": 1,
                    "doi": "https://doi.org/10.25405/data.ncl.24551539.v1"},
    "Gutenberg":   {"host": "ncl", "ncl_id": 24574753, "version": 1,
                    "doi": "https://doi.org/10.25405/data.ncl.24574753.v1"},
    "GeoClassing": {"host": "ncl", "ncl_id": 24050256, "version": 3,
                    "doi": "https://doi.org/10.25405/data.ncl.24050256.v3"},
    "Chesseract":  {"host": "ncl", "ncl_id": 24118743, "version": 2,
                    "doi": "https://doi.org/10.25405/data.ncl.24118743.v2"},
    "Sudoku":      {"host": "ncl", "ncl_id": 26976121, "version": 1,
                    "doi": "https://doi.org/10.25405/data.ncl.26976121.v1"},
    "Voxel":       {"host": "ncl", "ncl_id": 26970223, "version": 1,
                    "doi": "https://doi.org/10.25405/data.ncl.26970223.v1"},
    "Myofibre":    {"host": "ncl", "ncl_id": 26969998, "version": 1,
                    "doi": "https://doi.org/10.25405/data.ncl.26969998.v1"},
    "GameOfLife":  {"host": "ncl", "ncl_id": 30000835, "version": 1,
                    "doi": "https://doi.org/10.25405/data.ncl.30000835"},
    "Cryptic":     {"host": "edinburgh", "ncl_id": None, "version": 1,
                    "doi": "https://doi.org/10.7488/ds/8054"},
    "Windspeed":   {"host": "edinburgh", "ncl_id": None, "version": 1,
                    "doi": "https://doi.org/10.7488/ds/8053"},
}

NCL_ZIP_URL = "https://data.ncl.ac.uk/ndownloader/articles/{ncl_id}/versions/{version}"

REQUIRED_FILES = {
    "train_x.npy", "train_y.npy",
    "valid_x.npy", "valid_y.npy",
    "test_x.npy",  "test_y.npy",
    "metadata",
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _make_session():
    try:
        import requests
        s = requests.Session()
        s.headers["User-Agent"] = "NAS-Competition-Downloader/1.0"
        return s, "requests"
    except ImportError:
        return None, "urllib"


def _download_bytes(url, session, label=""):
    """Download url → bytes with a live progress counter."""
    chunk = 1 << 20  # 1 MB chunks

    if session is not None:
        resp = session.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        buf, received = io.BytesIO(), 0
        for data in resp.iter_content(chunk_size=chunk):
            buf.write(data)
            received += len(data)
            _progress(label, received, total)
        print()
        return buf.getvalue()

    else:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "NAS-Competition-Downloader/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            buf, received = io.BytesIO(), 0
            while True:
                data = resp.read(chunk)
                if not data:
                    break
                buf.write(data)
                received += len(data)
                _progress(label, received, total)
            print()
            return buf.getvalue()


def _progress(label, received, total):
    mb = received / (1 << 20)
    if total:
        pct = received / total * 100
        bar_w = 30
        filled = int(bar_w * received / total)
        bar = "█" * filled + "░" * (bar_w - filled)
        print(f"\r  {label}: [{bar}] {pct:5.1f}%  {mb:6.1f} MB", end="", flush=True)
    else:
        print(f"\r  {label}: {mb:6.1f} MB downloaded…", end="", flush=True)


# ---------------------------------------------------------------------------
# Per-host download strategies
# ---------------------------------------------------------------------------

def _download_ncl(name, info, dest_dir, session):
    url = NCL_ZIP_URL.format(ncl_id=info["ncl_id"], version=info["version"])
    print(f"  URL: {url}")
    raw = _download_bytes(url, session, label=name)

    # Try as ZIP first; some NCL articles serve individual files instead
    if zipfile.is_zipfile(io.BytesIO(raw)):
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            members = zf.namelist()
            for member in members:
                fname = Path(member).name
                if fname in REQUIRED_FILES or fname.endswith(".npy"):
                    zf.extract(member, dest_dir)
                    extracted = dest_dir / member
                    final = dest_dir / fname
                    if extracted != final:
                        final.parent.mkdir(parents=True, exist_ok=True)
                        extracted.rename(final)

        # Clean up any subdirectories left behind by the ZIP
        for item in dest_dir.iterdir():
            if item.is_dir():
                for sub in item.iterdir():
                    sub.rename(dest_dir / sub.name)
                try:
                    item.rmdir()
                except OSError:
                    pass
    else:
        # Not a zip — fall back to figshare v2 API to fetch individual files
        print(f"  Not a zip — trying figshare v2 API for individual files…")
        _download_ncl_via_api(name, info, dest_dir, session)


def _download_ncl_via_api(name, info, dest_dir, session):
    """Download individual dataset files using the figshare v2 REST API."""
    api_url = f"https://api.figshare.com/v2/articles/{info['ncl_id']}/files"
    print(f"  API: {api_url}")

    try:
        raw_meta = _download_bytes(api_url, session, label=f"{name}-api")
        files_meta = json.loads(raw_meta)
    except Exception as e:
        raise RuntimeError(f"figshare API call failed: {e}")

    if not isinstance(files_meta, list) or len(files_meta) == 0:
        raise RuntimeError("figshare API returned no files")

    downloaded = 0
    for file_info in files_meta:
        fname = file_info.get("name", "")
        dl_url = file_info.get("download_url", "")
        if not dl_url:
            dl_url = file_info.get("downloadUrl", "")
        if fname in REQUIRED_FILES or fname.endswith(".npy"):
            print(f"  Downloading file: {fname}")
            data = _download_bytes(dl_url, session, label=fname)
            (dest_dir / fname).write_bytes(data)
            downloaded += 1

    if downloaded == 0:
        raise RuntimeError(
            f"figshare API listed {len(files_meta)} files but none matched required names: "
            f"{[f.get('name') for f in files_meta]}"
        )


def _download_edinburgh(name, info, dest_dir, session):
    """
    Edinburgh DataShare uses DSpace. We follow the DOI redirect to find the
    dataset page, then look for a zip download link.
    """
    doi_url = info["doi"]
    print(f"  Following DOI: {doi_url}")

    try:
        if session is not None:
            resp = session.get(doi_url, timeout=30, allow_redirects=True)
            resp.raise_for_status()
            final_url = resp.url
            html = resp.text
        else:
            import urllib.request
            req = urllib.request.Request(
                doi_url, headers={"User-Agent": "NAS-Competition-Downloader/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                final_url = r.url
                html = r.read().decode("utf-8", errors="replace")
    except Exception as e:
        _manual_fallback(name, info, str(e))
        return False

    # Look for a direct .zip download link in the HTML
    import re
    zip_links = re.findall(r'href="([^"]+\.zip)"', html)
    if not zip_links:
        zip_links = re.findall(r'href="([^"]+/bitstream/[^"]+)"', html)

    if not zip_links:
        _manual_fallback(name, info, "could not find zip link on page")
        return False

    from urllib.parse import urljoin
    zip_url = urljoin(final_url, zip_links[0])
    print(f"  Found zip: {zip_url}")
    raw = _download_bytes(zip_url, session, label=name)

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for member in zf.namelist():
            fname = Path(member).name
            if fname in REQUIRED_FILES or fname.endswith(".npy"):
                data = zf.read(member)
                (dest_dir / fname).write_bytes(data)

    return True


def _manual_fallback(name, info, reason):
    print(f"\n  ⚠  Could not auto-download {name}: {reason}")
    print(f"     Please download manually from: {info['doi']}")
    print(f"     Then unzip into:  datasets/{name}/")
    print(f"     Required files: {', '.join(sorted(REQUIRED_FILES))}\n")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _sanitize_metadata_json(raw: str) -> str:
    """
    Fix known invalid JSON patterns that appear in some competition metadata files.

    Observed case: Edinburgh DataShare datasets (Cryptic, Windspeed) ship with
    bare unquoted ? as a value, e.g. "benchmark": ?  — valid Python/YAML but
    invalid JSON. json.loads raises JSONDecodeError: Expecting value.

    We replace every occurrence of  : <optional-whitespace> ?
    followed by , } or end-of-line with : null so the file round-trips cleanly.
    """
    # bare ? value:  "key": ?  ->  "key": null
    return re.sub(r':\s*\?(\s*[,}\]\n\r])', r': null\1', raw)


def _verify(name, dataset_dir, time_limit_hours=None):
    missing = REQUIRED_FILES - {f.name for f in dataset_dir.iterdir()}
    if missing:
        print(f"  ⚠  {name}: missing files: {', '.join(sorted(missing))}")
        return False

    meta_path = dataset_dir / "metadata"
    try:
        # read_bytes + decode strips UTF-8 BOM if present; sanitize fixes
        # known invalid JSON patterns (e.g. bare ? values in Edinburgh datasets).
        raw = meta_path.read_bytes().decode("utf-8-sig").strip()
        raw = _sanitize_metadata_json(raw)
        meta = json.loads(raw)
        needed = {"num_classes", "input_shape", "codename"}
        if not needed.issubset(meta.keys()):
            print(f"  ⚠  {name}: metadata missing keys: {needed - meta.keys()}")
            return False

    except Exception as e:
        print(f"  ⚠  {name}: metadata unreadable: {e}")
        return False

    # Always write time_limit into the metadata JSON so the competition runner picks it up.
    # Re-serialise through json.dumps to normalise encoding, strip BOM, and
    # guarantee a clean round-trip for every downstream json.load() call.
    if time_limit_hours is not None:
        meta['time_limit'] = time_limit_hours
    elif 'time_limit' not in meta:
        meta['time_limit'] = 0.5
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')

    tl = meta['time_limit']
    print(f"  ✓  {name}: OK  (codename={meta['codename']!r}"
          f"  classes={meta['num_classes']}  shape={meta['input_shape']}"
          f"  time_limit={tl}h)")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download NAS competition practice datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("datasets", nargs="*",
                        help="Dataset names to download (default: all).")
    parser.add_argument("--list", action="store_true",
                        help="List available datasets and exit.")
    parser.add_argument("--out", default="datasets",
                        help="Output directory (default: datasets/).")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip datasets that already look complete.")
    parser.add_argument("--time-limit", type=float, default=None, metavar="HOURS",
                        help="Write this time_limit into each dataset's metadata JSON. "
                             "Default: keep existing value or set 0.5h if absent.")
    parser.add_argument("--patch-time-limit", action="store_true",
                        help="Only patch time_limit in existing datasets (no download).")
    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for name, info in DATASETS.items():
            print(f"  {name:<14} {info['doi']}")
        return

    targets = args.datasets if args.datasets else list(DATASETS.keys())
    unknown = set(targets) - set(DATASETS.keys())
    if unknown:
        print(f"Unknown datasets: {', '.join(sorted(unknown))}")
        print(f"Valid names: {', '.join(DATASETS.keys())}")
        sys.exit(1)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    tl = args.time_limit  # may be None → _verify keeps existing or sets 0.5h

    # ── patch-only mode: just rewrite time_limit in existing metadata ──────────
    if args.patch_time_limit:
        print(f"Patching time_limit={tl if tl is not None else '(keep/default 0.5h)'} in existing datasets\n")
        for name in targets:
            dest = out_root / name
            if not dest.exists() or not (dest / "metadata").exists():
                print(f"  ⚠  {name}: not found at {dest}")
                continue
            _verify(name, dest, time_limit_hours=tl)
        return

    session, backend = _make_session()
    print(f"HTTP backend: {backend}")
    print(f"Output dir:   {out_root.resolve()}\n")

    results = {}
    for name in targets:
        info = DATASETS[name]
        dest = out_root / name
        dest.mkdir(parents=True, exist_ok=True)

        # Skip if already complete
        if args.skip_existing:
            existing = {f.name for f in dest.iterdir()} if dest.exists() else set()
            if REQUIRED_FILES.issubset(existing):
                print(f"[{name}] already present — patching time_limit and skipping re-download.")
                results[name] = _verify(name, dest, time_limit_hours=tl)
                continue

        print(f"\n[{name}]  ({info['host'].upper()}  DOI: {info['doi']})")
        t0 = time.time()
        try:
            if info["host"] == "ncl":
                _download_ncl(name, info, dest, session)
                ok = _verify(name, dest, time_limit_hours=tl)
            else:
                ok = _download_edinburgh(name, info, dest, session)
                if ok:
                    ok = _verify(name, dest, time_limit_hours=tl)
        except Exception as e:
            _manual_fallback(name, info, str(e))
            ok = False

        elapsed = time.time() - t0
        results[name] = ok
        status = "✓" if ok else "✗"
        print(f"  {status} {name} in {elapsed:.1f}s")

    # Summary
    print("\n" + "=" * 50)
    print("Summary:")
    ok_count = sum(results.values())
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'}  {name}")
    print(f"\n{ok_count}/{len(results)} datasets ready.")
    if ok_count < len(results):
        print("\nFor failed datasets, download manually from the DOI links above")
        print("and unzip into datasets/<DatasetName>/ with the required files.")


if __name__ == "__main__":
    main()
