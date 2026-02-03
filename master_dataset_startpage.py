#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, io, re, csv, json, gzip, bz2, lzma, tarfile, zipfile, zlib, argparse, sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional

csv.field_size_limit(sys.maxsize)
# ---------- optional codecs ----------
try:
    import zstandard as zstd
except Exception:
    zstd = None

# prefer brotlicffi; fallback to brotli; patch old brotli
_br = None
try:
    import brotlicffi as _br
except Exception:
    try:
        import brotli as _br
    except Exception:
        _br = None
if _br is not None and _br.__name__ == "brotli":
    try:
        if not hasattr(_br.Decompressor, "unused_data"):
            _br.Decompressor.unused_data = property(lambda self: b"")
    except Exception:
        pass

try:
    import lz4.frame as lz4f
except Exception:
    lz4f = None

from warcio.archiveiterator import ArchiveIterator
from resiliparse.parse.html import HTMLTree
from resiliparse.extract.html2text import extract_plain_text

ENGINE_TOKEN = "startpage"

HTML_EXTS = (".html", ".htm", ".xhtml")
WARC_LIKE_EXTS = (
    ".warc", ".warc.gz", ".warc.bz2", ".warc.xz", ".warc.zst", ".warc.br", ".warc.lz4",
    ".warc.wet.gz", ".wet.gz",
    ".arc", ".arc.gz"
)
TAR_LIKE_EXTS  = (".tar", ".tar.gz", ".tar.xz", ".tar.zst", ".tar.lz4", ".tgz", ".txz")
ZIP_EXTS       = (".zip",)

SNIFF_MAX_BYTES = 2 * 1024 * 1024
SNIFF_HEAD = 8192
DATE_MIN = None
ONLY_DATES = None  # set of 'YYYY-MM-DD' strings, or None

# date like 2023-04-11 anywhere in a path segment
DATE_SEG_RE  = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b", re.I)
# crawl id segment e.g. "2023-04-11-startpage"
CRAWL_SEG_RE = re.compile(r"20\d{2}-\d{2}-\d{2}[^/\\]*startpage", re.I)

# -------- asset filtering (avoid JS/CSS/JSON/images/fonts/etc.) --------
ASSET_EXTS = {
    ".js",".mjs",".css",".map",".json",".xml",".rss",".atom",
    ".ico",".svg",
    ".jpg",".jpeg",".png",".gif",".bmp",".webp",
    ".woff",".woff2",".ttf",".otf",".eot",
    ".pdf",".zip",".tar",".gz",".bz2",".xz",".zst",".7z",
    ".mp3",".mp4",".webm",".mkv",".avi",".mov",".wav",".ogg",
}
NON_HTML_MIME_BAD = ("javascript","json","xml","font","octet-stream","image","css","pdf","zip","gzip")
_ASSET_URL_HINTS = (
    "/client/static/", "/static/js/", "/static/css/", "/fonts/", "/font/", "/assets/",
    "/af/", "/favicon", "/favicons", "/sprite", "/glyphs"
)

def is_asset_url(url: str) -> bool:
    try:
        from urllib.parse import urlparse
        ext = os.path.splitext(urlparse(url).path.lower())[1]
        return ext in ASSET_EXTS
    except Exception:
        return False

def looks_like_asset_url_heuristic(url: str) -> bool:
    u = (url or "").lower()
    return any(hint in u for hint in _ASSET_URL_HINTS)

def is_asset_path(name: str) -> bool:
    ext = os.path.splitext(name.lower().split("::")[-1])[1]
    return ext in ASSET_EXTS

def is_disqualifying_mime(ctype: str) -> bool:
    c = (ctype or "").lower()
    return any(tok in c for tok in NON_HTML_MIME_BAD)

# ---------- crawl/folder helpers ----------
def crawl_id_from_path(path: str) -> str:
    parts = Path(path).parts
    for seg in parts:
        if CRAWL_SEG_RE.search(seg or ""):
            return seg
    for seg in reversed(parts):
        if "startpage" in (seg or "").lower():
            return seg
    return "unknown-startpage"

def folder_key_from_source(kind: str, path_or_disp: str) -> str:
    """
    Stable folder id:
      - FS file: its directory
      - Archive member: 'outer.tar...::dirname(inner/path)'
      - WARC member stream: same as above
    """
    if "::" in path_or_disp:
        outer, inner = path_or_disp.split("::", 1)
        inner_dir = os.path.dirname(inner) or "."
        return f"{outer}::{inner_dir}"
    return os.path.dirname(path_or_disp) or "."

def is_engine_dir(root: str, dirpath: str, token: str) -> bool:
    rel = os.path.relpath(dirpath, root)
    return any(token in p.lower() for p in Path(rel).parts)

# ---------- robust HTML detection ----------
def contains_html_tag_outside_quotes(buf: bytes, head_len: int = SNIFF_HEAD) -> bool:
    head = (buf[:head_len] or b"")
    in_sq = False
    in_dq = False
    i = 0
    while i < len(head):
        c = head[i:i+1]
        if c == b"\\":
            i += 2; continue
        if not in_dq and c == b"'":
            in_sq = not in_sq; i += 1; continue
        if not in_sq and c == b'"':
            in_dq = not in_dq; i += 1; continue
        if not in_sq and not in_dq and c == b"<":
            nxt = head[i+1:i+9].lower()
            if nxt.startswith(b"!doctype") or nxt.startswith(b"html"):
                return True
            if i + 2 < len(head):
                c1 = head[i+1:i+2]
                if b"a" <= c1 <= b"z":
                    return True
        i += 1
    return False

def is_binary_like(buf: bytes, sample: int = 65536) -> bool:
    s = buf[:sample]
    if not s:
        return False
    if b"\x00" in s:
        return True
    text_chars = set([7,8,9,10,12,13,27] + list(range(32,127)))
    nontext = sum(b not in text_chars for b in s)
    return (nontext / max(1, len(s))) > 0.30

def is_probably_html(content_type: str, raw: bytes) -> bool:
    ctype = (content_type or "").lower()
    if "html" in ctype:
        return not is_binary_like(raw)
    if is_disqualifying_mime(ctype):
        return False
    return contains_html_tag_outside_quotes(raw) and not is_binary_like(raw)

# ---------- HTML extraction ----------
def extract_main(html_bytes_or_str) -> str:
    html_str = (html_bytes_or_str.decode("utf-8", errors="ignore")
                if isinstance(html_bytes_or_str, (bytes, bytearray))
                else str(html_bytes_or_str))
    tree = HTMLTree.parse(html_str)
    text = extract_plain_text(tree, preserve_formatting=False)
    return " ".join(text.strip().split())

def _brotli_decompress(data: bytes) -> bytes:
    if _br is None:
        raise RuntimeError("brotli unavailable")
    try:
        d = _br.Decompressor()
        return d.decompress(data)
    except Exception:
        return _br.decompress(data)

def decompress_body(body: bytes, content_encoding: Optional[str]) -> tuple[bytes, bool]:
    if not body:
        return body, True
    enc = (content_encoding or "").lower()
    try:
        if "br" in enc:
            return _brotli_decompress(body), True
        if "zstd" in enc or "zstandard" in enc:
            if zstd is not None:
                return zstd.ZstdDecompressor().decompress(body), True
        if "gzip" in enc or "x-gzip" in enc:
            try:
                return gzip.decompress(body), True
            except Exception:
                out = bytearray(); bio = io.BytesIO(body)
                while True:
                    try:
                        with gzip.GzipFile(fileobj=bio) as gzf:
                            out.extend(gzf.read())
                    except OSError:
                        break
                    if bio.tell() >= len(body):
                        break
                if out:
                    return bytes(out), True
        if "deflate" in enc:
            try:
                return zlib.decompress(body, -zlib.MAX_WBITS), True
            except zlib.error:
                try:
                    return zlib.decompress(body), True
                except Exception:
                    pass
        if "lz4" in enc and lz4f is not None:
            return lz4f.decompress(body), True
    except Exception:
        pass
    if contains_html_tag_outside_quotes(body) and not is_binary_like(body):
        return body, True
    return body, False

def open_compressed_stream(path: str):
    lower = path.lower()
    if lower.endswith(".gz"):
        return gzip.open(path, "rb")
    if lower.endswith(".bz2"):
        return bz2.open(path, "rb")
    if lower.endswith(".xz"):
        return lzma.open(path, "rb")
    if lower.endswith(".zst") and zstd is not None:
        base = open(path, "rb")
        return zstd.ZstdDecompressor().stream_reader(base)
    if lower.endswith(".br"):
        raw = open(path, "rb").read()
        data = _brotli_decompress(raw) if _br else raw
        return io.BytesIO(data)
    if lower.endswith(".lz4") and lz4f is not None:
        data = lz4f.decompress(open(path, "rb").read())
        return io.BytesIO(data)
    return open(path, "rb")

def open_tar_stream_from_fileobj(fobj):
    return tarfile.open(fileobj=fobj, mode="r|*")  # streaming tar reader

def read_text_file(p: Path) -> Optional[str]:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

def parse_json_file(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

def guess_url_from_html(html_str: str) -> str:
    m = re.search(r'rel\s*=\s*["\']canonical["\'][^>]*href\s*=\s*["\']([^"\']+)["\']', html_str, re.I)
    if m and m.group(1).startswith("http"): return m.group(1)
    m = re.search(r'<meta[^>]+property\s*=\s*["\']og:url["\'][^>]*content\s*=\s*["\']([^"\']+)["\']', html_str, re.I)
    if m and m.group(1).startswith("http"): return m.group(1)
    m = re.search(r'<base[^>]+href\s*=\s*["\']([^"\']+)["\']', html_str, re.I)
    if m and m.group(1).startswith("http"): return m.group(1)
    m = re.search(r'<meta[^>]+itemprop\s*=\s*["\']url["\'][^>]*content\s*=\s*["\']([^"\']+)["\']', html_str, re.I)
    if m and m.group(1).startswith("http"): return m.group(1)
    return ""

def guess_url_from_neighbors_fs(html_path: Path, html_str: str) -> str:
    snap_dir = html_path.parent
    t = read_text_file(snap_dir / "url.txt")
    if t and t.strip().startswith("http"): return t.strip()
    for name in ("meta.json","pageinfo.json","info.json","request.json"):
        j = parse_json_file(snap_dir / name)
        if j:
            for k in ("url","final_url","requested_url","page_url"):
                v = j.get(k)
                if isinstance(v, str) and v.startswith("http"): return v
    t = read_text_file(snap_dir / "request.txt")
    if t:
        m = re.search(r"^\s*(GET|POST)\s+(\S+)\s+HTTP", t, flags=re.I|re.M)
        if m and m.group(2).startswith("http"): return m.group(2)
    t = read_text_file(snap_dir / "response-headers.txt")
    if t:
        m = re.search(r"(?mi)^(?:content-location|location):\s*(\S+)", t)
        if m and m.group(1).startswith("http"): return m.group(1).strip()
    return guess_url_from_html(html_str)

# ---------- pruning helpers ----------
def path_date_if_any(p: str) -> Optional[str]:
    m = DATE_SEG_RE.search(p or "")
    return m.group(1) if m else None

def should_skip_path(p: str, done_dates: set[str]) -> bool:
    d = path_date_if_any(p)
    if d:
        if ONLY_DATES is not None and d not in ONLY_DATES:
            return True
        if d in done_dates:
            return True
        if DATE_MIN and d < DATE_MIN:
            return True
    return False

# ---------- iterators over inputs (with pruning) ----------
def iter_from_tar(display_prefix: str, tf: tarfile.TarFile, done_dates: set[str]):
    for m in tf:
        if not m.isreg(): continue
        name = m.name
        disp = f"{display_prefix}::{name}"
        if should_skip_path(disp, done_dates):
            continue
        lower = name.lower()
        try:
            fobj = tf.extractfile(m)
        except Exception:
            fobj = None
        if fobj is None: continue

        if lower.endswith(TAR_LIKE_EXTS):
            try:
                tf2 = open_tar_stream_from_fileobj(fobj)
            except Exception:
                continue
            yield from iter_from_tar(disp, tf2, done_dates)
            try: tf2.close()
            except Exception: pass
            continue

        if lower.endswith(ZIP_EXTS):
            try:
                with zipfile.ZipFile(fobj) as zf:
                    for item in iter_from_zip(disp, zf, done_dates):
                        yield item
            except Exception:
                pass
            continue

        if lower.endswith(WARC_LIKE_EXTS):
            try:
                if lower.endswith(".gz"):
                    stream = gzip.GzipFile(fileobj=fobj)
                elif lower.endswith(".bz2"):
                    stream = bz2.BZ2File(fobj)
                elif lower.endswith(".xz"):
                    stream = lzma.LZMAFile(fobj)
                elif lower.endswith(".zst") and zstd is not None:
                    stream = zstd.ZstdDecompressor().stream_reader(fobj)
                elif lower.endswith(".br"):
                    stream = io.BytesIO(_brotli_decompress(fobj.read()) if _br else fobj.read())
                elif lower.endswith(".lz4") and lz4f is not None:
                    stream = io.BytesIO(lz4f.decompress(fobj.read()))
                else:
                    stream = fobj
                yield ("warc", disp, stream)
            except Exception:
                pass
            continue

        try:
            if lower.endswith(HTML_EXTS) or m.size <= SNIFF_MAX_BYTES:
                data = fobj.read()
                if lower.endswith(HTML_EXTS) or contains_html_tag_outside_quotes(data):
                    yield ("html-bytes", disp, data)
        except Exception:
            pass

def iter_from_zip(display_prefix: str, zf: zipfile.ZipFile, done_dates: set[str]):
    for name in zf.namelist():
        disp = f"{display_prefix}::{name}"
        if should_skip_path(disp, done_dates):
            continue
        lower = name.lower()

        if lower.endswith(TAR_LIKE_EXTS):
            try:
                with zf.open(name) as f:
                    tf2 = open_tar_stream_from_fileobj(f)
                    for item in iter_from_tar(disp, tf2, done_dates):
                        yield item
                    try: tf2.close()
                    except Exception: pass
            except Exception:
                pass
            continue

        if lower.endswith(ZIP_EXTS):
            try:
                with zf.open(name) as f:
                    with zipfile.ZipFile(f) as zf2:
                        for item in iter_from_zip(disp, zf2, done_dates):
                            yield item
            except Exception:
                pass
            continue

        if lower.endswith(WARC_LIKE_EXTS):
            try:
                stream = zf.open(name)
                yield ("warc", disp, stream)
            except Exception:
                pass
            continue

        try:
            info = zf.getinfo(name)
            if lower.endswith(HTML_EXTS) or info.file_size <= SNIFF_MAX_BYTES:
                with zf.open(name) as f:
                    data = f.read() if lower.endswith(HTML_EXTS) else f.read(SNIFF_MAX_BYTES + 1)
                    if lower.endswith(HTML_EXTS) or contains_html_tag_outside_quotes(data):
                        if not lower.endswith(HTML_EXTS) and info.file_size > len(data):
                            with zf.open(name) as f2:
                                data = f2.read()
                        yield ("html-bytes", disp, data)
        except Exception:
            pass

def iter_all_inputs(root: str, engine_token: str, done_dates: set[str]):
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        if not is_engine_dir(root, dirpath, engine_token):
            continue

        if should_skip_path(dirpath, done_dates):
            dirnames[:] = []
            continue

        # prune child dirs already done
        dirnames[:] = [d for d in dirnames if not should_skip_path(os.path.join(dirpath, d), done_dates)]

        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            if should_skip_path(fp, done_dates):
                continue

            lower = fn.lower()

            if lower.endswith(HTML_EXTS):
                yield ("html-fs", fp, None); continue
            try:
                if not (lower.endswith(WARC_LIKE_EXTS) or lower.endswith(TAR_LIKE_EXTS) or lower.endswith(ZIP_EXTS)):
                    if os.path.getsize(fp) <= SNIFF_MAX_BYTES:
                        with open(fp, "rb") as f:
                            head = f.read(SNIFF_HEAD)
                        if contains_html_tag_outside_quotes(head) and not is_binary_like(head):
                            yield ("html-fs", fp, None); continue
            except Exception:
                pass

            if lower.endswith(WARC_LIKE_EXTS):
                try:
                    yield ("warc", fp, open_compressed_stream(fp))
                except Exception as e:
                    print(f"⚠️  Cannot open {fp}: {e}")
                continue

            if lower.endswith(TAR_LIKE_EXTS):
                try:
                    if lower.endswith(".tar.zst"):
                        if zstd is None:
                            print(f"⚠️  Skipping {fp}: zstandard not installed"); continue
                        base = open(fp, "rb")
                        stream = zstd.ZstdDecompressor().stream_reader(base)
                        tf = tarfile.open(fileobj=stream, mode="r|*")
                        try:
                            for item in iter_from_tar(fp, tf, done_dates):
                                yield item
                        finally:
                            for obj in (tf, stream, base):
                                try: obj.close()
                                except Exception: pass
                    elif lower.endswith(".tar.lz4") and lz4f is not None:
                        raw = open(fp, "rb").read()
                        data = lz4f.decompress(raw)
                        tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:*")
                        try:
                            for item in iter_from_tar(fp, tf, done_dates):
                                yield item
                        finally:
                            try: tf.close()
                            except Exception: pass
                    else:
                        with tarfile.open(fp, mode="r|*") as tf:
                            for item in iter_from_tar(fp, tf, done_dates):
                                yield item
                except Exception as e:
                    print(f"⚠️  Could not read TAR container: {fp} ({e})")
                continue

            if lower.endswith(ZIP_EXTS):
                try:
                    with zipfile.ZipFile(fp) as zf:
                        for item in iter_from_zip(fp, zf, done_dates):
                            yield item
                except Exception as e:
                    print(f"⚠️  Could not read ZIP container: {fp} ({e})")
                continue

# ---------- selection + writing ----------
def write_row(writer, row):
    writer.writerow(row)
    writer._file.flush()   # type: ignore[attr-defined]
    os.fsync(writer._file.fileno())  # type: ignore[attr-defined]

def add_if_needed(date_str: Optional[str], url: str, ts: str, display_path: str, text: str,
                  writer, stats: Counter,
                  done_dates: set, per_date_counts: dict,
                  seen_url_by_tuple: set,
                  seen_by_crawl: dict, crawl_id: str, folder_key: str,
                  per_date_k: int) -> None:
    """
    Enforce:
      - stop after K per date,
      - unique URL per (date, folder, crawl),
      - dedupe per crawl,
      - append to CSV immediately.
    """
    if not date_str or date_str in done_dates:
        return

    # per-crawl dedupe
    key_crawl = (crawl_id, url.lower())
    if key_crawl in seen_by_crawl[crawl_id]:
        return

    # unique per (date, folder, crawl)
    key_tuple = (date_str, crawl_id, folder_key, url.lower())
    if key_tuple in seen_url_by_tuple:
        return

    # accept
    seen_by_crawl[crawl_id].add(key_crawl)
    seen_url_by_tuple.add(key_tuple)

    write_row(writer, {
        "Date": date_str,  
        "URL": url,
        "Timestamp": ts,
        "File Path": display_path,
        "Text": text
    })
    stats["written"] += 1
    per_date_counts[date_str] += 1

    if per_date_counts[date_str] >= per_date_k:
        done_dates.add(date_str)

# ---------- processors (with early exit if date is done; WARC mid-stream bail) ----------
def process_warc_stream(stream, display_path: str, writer, stats: Counter,
                        done_dates, per_date_counts, seen_url_by_tuple, seen_by_crawl,
                        per_date_k: int):
    if should_skip_path(display_path, done_dates):
        try: stream.close()
        except Exception: pass
        return

    crawl_id  = crawl_id_from_path(display_path)
    folder_key = folder_key_from_source("warc", display_path)

    # date from WARC path is constant across its records
    m = DATE_SEG_RE.search(display_path)
    date_from_path = m.group(1) if m else None

    try:
        for rec in ArchiveIterator(stream):
            # bail mid-stream if date became done
            if date_from_path and date_from_path in done_dates:
                break

            try:
                if rec.rec_type != "response":
                    continue
                url = (rec.rec_headers.get_header("WARC-Target-URI") or "").strip()
                ts  = rec.rec_headers.get_header("WARC-Date") or ""

                if url and (is_asset_url(url) or looks_like_asset_url_heuristic(url)):
                    continue

                http = rec.http_headers
                if not http:
                    continue

                status = 0
                try:
                    if hasattr(http, "get_statuscode"):
                        status = int(http.get_statuscode())
                    elif getattr(http, "statusline", None):
                        parts = (http.statusline or "").split()
                        if len(parts) >= 2 and parts[1].isdigit():
                            status = int(parts[1])
                except Exception:
                    status = 0
                if not (200 <= status <= 299):
                    continue

                ctype = http.get_header("Content-Type") or ""
                cenc  = http.get_header("Content-Encoding") or ""
                if "html" not in ctype.lower():
                    continue

                raw = rec.content_stream().read()
                raw, dec_ok = decompress_body(raw, cenc)
                if not dec_ok or is_binary_like(raw):
                    continue
                if not contains_html_tag_outside_quotes(raw):
                    continue

                text = extract_main(raw)
                if not (url and text):
                    continue

                # use date from path for speed/consistency
                date_str = date_from_path

                add_if_needed(date_str, url, ts, display_path, text,
                              writer, stats,
                              done_dates, per_date_counts,
                              seen_url_by_tuple,
                              seen_by_crawl, crawl_id, folder_key,
                              per_date_k)

                # if we just completed this date, stop reading this WARC right away
                if date_str and date_str in done_dates:
                    break

            except Exception:
                stats["record_error"] += 1
    except Exception as e:
        stats["file_error"] += 1
        print(f"⚠️  File error: {display_path} → {e}")
    finally:
        try: stream.close()
        except Exception: pass

def process_html_fs(path: str, writer, stats: Counter,
                    done_dates, per_date_counts, seen_url_by_tuple, seen_by_crawl,
                    per_date_k: int):
    if should_skip_path(path, done_dates):
        return
    if is_asset_path(path) or looks_like_asset_url_heuristic(path):
        return
    crawl_id = crawl_id_from_path(path)
    folder_key = folder_key_from_source("html-fs", path)

    try:
        b = open(path, "rb").read()
    except Exception:
        stats["file_error"] += 1
        return

    if not is_probably_html("", b):
        return
    text = extract_main(b)
    if not text:
        return

    html_str = b.decode("utf-8", errors="ignore")
    url = (guess_url_from_neighbors_fs(Path(path), html_str) or "").strip()
    if not url:
        return

    m = DATE_SEG_RE.search(str(path))
    date_str = m.group(1) if m else None
    ts = (date_str + "T00:00:00Z") if date_str else ""

    add_if_needed(date_str, url, ts, str(path), text,
                  writer, stats,
                  done_dates, per_date_counts,
                  seen_url_by_tuple,
                  seen_by_crawl, crawl_id, folder_key,
                  per_date_k)

def process_html_bytes(display: str, html_bytes: bytes, writer, stats: Counter,
                       done_dates, per_date_counts, seen_url_by_tuple, seen_by_crawl,
                       per_date_k: int):
    if should_skip_path(display, done_dates):
        return
    if is_asset_path(display) or looks_like_asset_url_heuristic(display):
        return
    crawl_id = crawl_id_from_path(display)
    folder_key = folder_key_from_source("html-bytes", display)

    if not is_probably_html("", html_bytes):
        return
    text = extract_main(html_bytes)
    if not text:
        return

    url = (guess_url_from_html(html_bytes.decode("utf-8", errors="ignore")) or "").strip()
    if not url:
        return

    m = DATE_SEG_RE.search(display)
    date_str = m.group(1) if m else None
    ts = (date_str + "T00:00:00Z") if date_str else ""

    add_if_needed(date_str, url, ts, display, text,
                  writer, stats,
                  done_dates, per_date_counts,
                  seen_url_by_tuple,
                  seen_by_crawl, crawl_id, folder_key,
                  per_date_k)

def load_existing_state(out_csv: str, per_date_k: int,
                        done_dates: set[str],
                        per_date_counts: dict,
                        seen_url_by_tuple: set[tuple],
                        seen_by_crawl: dict):
    """
    Resume support: rebuild counters and dedupe sets from an existing CSV.
    - Marks dates as 'done' if they already hit per_date_k
    - Reconstructs (date, crawl_id, folder_key, url_lower) tuples to avoid duplicates
    """
    if not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0:
        return

    with open(out_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = (row.get("URL") or "").strip().lower()
            disp = (row.get("File Path") or "").strip()
            ts   = (row.get("Timestamp") or "").strip()
            date = (row.get("Date") or "").strip()

            # Fallbacks if Date is missing (older CSVs)
            if not date:
                date = path_date_if_any(disp) or path_date_if_any(ts) or ""
            if not (date and url and disp):
                continue

            # Count + mark done dates
            per_date_counts[date] += 1
            if per_date_counts[date] >= per_date_k:
                done_dates.add(date)

            # Rebuild dedupe keys
            crawl_id = crawl_id_from_path(disp)
            kind = "warc" if "::" in disp or any(ext in disp.lower() for ext in (".warc", ".arc")) else "html-fs"
            folder_key = folder_key_from_source(kind, disp)

            seen_by_crawl[crawl_id].add((crawl_id, url))
            seen_url_by_tuple.add((date, crawl_id, folder_key, url))

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Startpage extractor: walk directory (and archives), STRICT HTML only; "
                    "append up to K unique URLs per date with main text content, "
                    "and prune traversal once a date reaches K."
    )
    ap.add_argument("--root", required=True, help="Top directory to scan (only 'startpage' subtrees are traversed)")
    ap.add_argument("--out",  required=True, help="Output CSV to append to")
    ap.add_argument("--engine", default=ENGINE_TOKEN, help="Path token to limit traversal (default: startpage)")
    ap.add_argument("--per-date-k", type=int, default=10, help="How many URLs to keep per date (default: 10)")
    ap.add_argument("--resume-csv", action="append", default=[],
                help="One or more existing CSVs to preload state from (use multiple --resume-csv flags).")
    ap.add_argument("--date-min", default=None,
                help="Only traverse dates >= this (YYYY-MM-DD)")
    ap.add_argument("--only-dates", default=None,
                help="Comma-separated list of YYYY-MM-DD to process; skip all other dates.")
    

    args = ap.parse_args()
    # after parsing:
    global DATE_MIN, ONLY_DATES
    DATE_MIN = args.date_min
    ONLY_DATES = set(args.only_dates.split(",")) if args.only_dates else None

    # open CSV in append mode; write header if new/empty
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new_file = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    f_out = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=["Date","URL","Timestamp","File Path","Text"])
    writer._file = f_out
    if new_file:
        writer.writeheader(); f_out.flush(); os.fsync(f_out.fileno())

    stats = Counter()

   # state
    done_dates: set[str] = set()
    per_date_counts = defaultdict(int)
    seen_url_by_tuple: set[tuple] = set()
    seen_by_crawl = defaultdict(set)

    # NEW: rebuild state from existing CSV (resume)
    load_existing_state(args.out, args.per_date_k,
                        done_dates, per_date_counts,
                        seen_url_by_tuple, seen_by_crawl)
    # existing:
    # load_existing_state(args.out, args.per_date_k,
    #                     done_dates, per_date_counts,
    #                     seen_url_by_tuple, seen_by_crawl)

    # keep the line above if you already added it; then add this:
    for prev_csv in args.resume_csv:
        load_existing_state(prev_csv, args.per_date_k,
                            done_dates, per_date_counts,
                            seen_url_by_tuple, seen_by_crawl)

    try:
        for kind, path_or_disp, payload in iter_all_inputs(args.root, args.engine, done_dates):
            if should_skip_path(path_or_disp, done_dates):
                if kind == "warc" and payload:
                    try: payload.close()
                    except Exception: pass
                continue

            if kind == "warc":
                print(f"📁 warc: {path_or_disp}")
                process_warc_stream(payload, path_or_disp, writer, stats,
                                    done_dates, per_date_counts, seen_url_by_tuple, seen_by_crawl,
                                    args.per_date_k)
            elif kind == "html-fs":
                print(f"📄 html(fs): {path_or_disp}")
                process_html_fs(path_or_disp, writer, stats,
                                done_dates, per_date_counts, seen_url_by_tuple, seen_by_crawl,
                                args.per_date_k)
            else:  # html-bytes (inside archives)
                print(f"📄 html(arc): {path_or_disp}")
                process_html_bytes(path_or_disp, payload, writer, stats,
                                   done_dates, per_date_counts, seen_url_by_tuple, seen_by_crawl,
                                   args.per_date_k)
    except KeyboardInterrupt:
        print("\n⏹ Interrupted by user.", file=sys.stderr)
    finally:
        try: f_out.close()
        except Exception: pass

    # summary
    print("\n📊 Per-date written (cap per date = %d):" % args.per_date_k)
    for d in sorted(per_date_counts.keys()):
        print(f"{d} : {per_date_counts[d]}")
    print(f"\n✅ Appended rows to → {args.out}")
    print(f"ℹ️  Stats: {dict(stats)}")

if __name__ == "__main__":
    main()




#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from collections import defaultdict
# import pandas as pd

# # === Hardcoded paths (edit if needed) ===
# MASTER_CSV = "/mnt/ceph/storage/data-tmp/current/yili5634/master_dataset.csv"          # expects columns: URL, Timestamp
# PREDS_CSV  = "/mnt/ceph/storage/data-tmp/current/yili5634/bert_predictions_binary_multiclass.csv"   # expects columns: url, timestamp

# CHUNKSIZE = 250_000  # tune if needed


# def find_col(cols, candidates):
#     for c in candidates:
#         if c in cols:
#             return c
#     return None


# def read_master_build_sets():
#     master_counts = defaultdict(int)
#     master_urls_by_date = defaultdict(set)
#     total_rows = 0
#     parsed_rows = 0

#     usecols = None  # detect first chunk
#     for chunk in pd.read_csv(MASTER_CSV, chunksize=CHUNKSIZE, dtype=str):
#         total_rows += len(chunk)
#         if usecols is None:
#             url_col = find_col(chunk.columns, ["URL", "url"])
#             ts_col  = find_col(chunk.columns, ["Timestamp", "timestamp"])
#             if not url_col or not ts_col:
#                 raise SystemExit(f"[MASTER] Missing URL/Timestamp columns. Found: {list(chunk.columns)}")
#             usecols = [url_col, ts_col]
#             # re-read this chunk only with needed cols
#             chunk = chunk[usecols]
#         else:
#             chunk = chunk[usecols]

#         ts = pd.to_datetime(chunk[usecols[1]], errors="coerce", utc=True)
#         day = ts.dt.floor("D")
#         mask = day.notna() & chunk[usecols[0]].notna()
#         parsed_rows += int(mask.sum())
#         if not mask.any():
#             continue

#         urls = chunk.loc[mask, usecols[0]].astype(str).str.strip().str.lower()
#         days = day[mask]

#         # counts per day
#         for d, cnt in days.value_counts().items():
#             master_counts[d] += int(cnt)

#         # URL sets per day (for overlap)
#         for d, u in zip(days, urls):
#             master_urls_by_date[d].add(u)

#     print(f"[MASTER] rows read: {total_rows:,} | parsed: {parsed_rows:,} | unique days: {len(master_counts):,}")
#     return master_counts, master_urls_by_date


# def read_preds_and_overlap(master_urls_by_date):
#     preds_counts = defaultdict(int)
#     overlap_counts = defaultdict(int)
#     total_rows = 0
#     parsed_rows = 0

#     usecols = None
#     for chunk in pd.read_csv(PREDS_CSV, chunksize=CHUNKSIZE, dtype=str):
#         total_rows += len(chunk)
#         if usecols is None:
#             url_col = find_col(chunk.columns, ["url", "URL"])
#             ts_col  = find_col(chunk.columns, ["timestamp", "Timestamp"])
#             if not url_col or not ts_col:
#                 raise SystemExit(f"[PREDS] Missing url/timestamp columns. Found: {list(chunk.columns)}")
#             usecols = [url_col, ts_col]
#             chunk = chunk[usecols]
#         else:
#             chunk = chunk[usecols]

#         ts = pd.to_datetime(chunk[usecols[1]], errors="coerce", utc=True)
#         day = ts.dt.floor("D")
#         mask = day.notna() & chunk[usecols[0]].notna()
#         parsed_rows += int(mask.sum())
#         if not mask.any():
#             continue

#         urls = chunk.loc[mask, usecols[0]].astype(str).str.strip().str.lower()
#         days = day[mask]

#         # count preds per day and compute overlap against master URL sets
#         for d, u in zip(days, urls):
#             preds_counts[d] += 1
#             if u in master_urls_by_date.get(d, ()):
#                 overlap_counts[d] += 1

#     print(f"[PREDS ] rows read: {total_rows:,} | parsed: {parsed_rows:,} | unique days: {len(preds_counts):,}")
#     return preds_counts, overlap_counts


# def main():
#     master_counts, master_urls_by_date = read_master_build_sets()
#     preds_counts, overlap_counts = read_preds_and_overlap(master_urls_by_date)

#     all_days = sorted(set(master_counts.keys()) | set(preds_counts.keys()))
#     print("\n== Per-day URL counts (master vs preds) + overlap ==")
#     print(f"{'date':<12} {'master':>10} {'preds':>10} {'overlap':>10} {'coverage':>10}")
#     print("-" * 58)

#     total_m = total_p = total_o = 0
#     for d in all_days:
#         m = master_counts.get(d, 0)
#         p = preds_counts.get(d, 0)
#         o = overlap_counts.get(d, 0)
#         cov = (p / m) if m else float("nan")
#         total_m += m
#         total_p += p
#         total_o += o
#         day_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
#         print(f"{day_str:<12} {m:>10,d} {p:>10,d} {o:>10,d} {cov:>10.2%}" if m else
#               f"{day_str:<12} {m:>10} {p:>10,d} {o:>10,d} {'-':>10}")

#     print("-" * 58)
#     overall_cov = (total_p / total_m) if total_m else float("nan")
#     print(f"{'TOTAL':<12} {total_m:>10,d} {total_p:>10,d} {total_o:>10,d} "
#           f"{(f'{overall_cov:.2%}' if total_m else '-'):>10}")

#     # Quick summaries
#     days_only_master = sum(1 for d in all_days if master_counts.get(d, 0) > 0 and preds_counts.get(d, 0) == 0)
#     days_only_preds  = sum(1 for d in all_days if preds_counts.get(d, 0) > 0 and master_counts.get(d, 0) == 0)
#     days_both        = sum(1 for d in all_days if master_counts.get(d, 0) > 0 and preds_counts.get(d, 0) > 0)
#     print(f"\nDays in master: {len([d for d in all_days if master_counts.get(d,0)>0]):,}  | "
#           f"Days in preds: {len([d for d in all_days if preds_counts.get(d,0)>0]):,}  | "
#           f"Days in both: {days_both:,}  | "
#           f"Only master: {days_only_master:,}  | Only preds: {days_only_preds:,}")


# if __name__ == "__main__":
#     main()
#--------------------------------------------------------------------------


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import re
# import sys
# from pathlib import Path
# from collections import Counter
# import pandas as pd

# # ====== CONFIG ======
# CSV_PATH   = "/mnt/ceph/storage/data-tmp/current/yili5634/startpage_10_per_date_v1.csv"
# CHUNKSIZE  = 200_000
# ENCODING   = "utf-8"
# SEP        = ","
# YEAR_MIN   = 2018
# YEAR_MAX   = 2025
# OUT_CSV    = "/mnt/ceph/storage/data-tmp/current/yili5634/startpage_counts_by_date.csv"

# # If you want to allow ONLY the official crawl days, uncomment:
# # ALLOWED_DATES = {"2022-08-24","2022-11-25","2023-03-27","2023-04-11"}
# ALLOWED_DATES = None

# # Column name hints (case-insensitive)
# URL_CANDIDATES  = ["url", "link", "page_url", "target_url", "source_url", "URL"]
# PATH_CANDIDATES = [
#     "file path","filepath","file_path","path","source","source_path","warc_path",
#     "snapshot","crawl_path","origin","archive","warc","warcfile","sourcefile","File Path"
# ]

# # Strict date regex with ONE capture group (whole date)
# DATE_RE = re.compile(r"\b((?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01]))\b")


# def infer_col(cols, candidates):
#     lower_map = {c.lower(): c for c in cols}
#     for cand in candidates:
#         if cand in lower_map:
#             return lower_map[cand]
#     return None


# def validate_dates_to_series(date_str_series, year_min=YEAR_MIN, year_max=YEAR_MAX):
#     s = pd.Series(date_str_series, copy=False)
#     dt = pd.to_datetime(s, errors="coerce", format="%Y-%m-%d")
#     ok = dt.notna() & dt.dt.year.between(year_min, year_max)
#     out = pd.Series(pd.NA, index=s.index, dtype="object")
#     out.loc[ok] = dt.loc[ok].dt.strftime("%Y-%m-%d")
#     return out


# def extract_dates_series(df: pd.DataFrame, path_col: str | None):
#     """Vectorized date extraction from path-like column, with row-join fallback."""
#     # First attempt: from a path-like column
#     if path_col and path_col in df.columns:
#         s_path = df[path_col].astype(str)
#         dates = s_path.str.extract(DATE_RE, expand=False)
#     else:
#         dates = pd.Series(pd.NA, index=df.index, dtype="object")

#     # Fallback: scan the whole row text for misses
#     missing = dates.isna()
#     if missing.any():
#         joined = df.loc[missing].astype(str).agg(" ".join, axis=1)
#         fb = joined.str.extract(DATE_RE, expand=False)
#         dates.loc[missing & fb.notna()] = fb[fb.notna()]

#     # Validate + optional whitelist
#     dates = validate_dates_to_series(dates, year_min=YEAR_MIN, year_max=YEAR_MAX)
#     if ALLOWED_DATES is not None:
#         dates = dates.where(dates.isin(ALLOWED_DATES), other=pd.NA)
#     return dates


# def main():
#     path = Path(CSV_PATH)
#     if not path.exists():
#         print(f"Error: file not found: {path}", file=sys.stderr)
#         sys.exit(1)

#     # Infer URL/path cols once
#     url_col = None
#     path_col = None
#     first_infer_done = False

#     total_urls = 0
#     per_date = Counter()
#     unknown_total = 0

#     reader = pd.read_csv(
#         path, sep=SEP, encoding=ENCODING, dtype=str,
#         chunksize=CHUNKSIZE, low_memory=True
#     )

#     for chunk in reader:
#         if not first_infer_done:
#             url_col  = infer_col(chunk.columns, URL_CANDIDATES)
#             path_col = infer_col(chunk.columns, PATH_CANDIDATES)
#             first_infer_done = True

#         # Heuristic if URL col not obvious
#         if url_col is None:
#             sample = chunk.head(2000)
#             best_col, best_hits = None, -1
#             for c in sample.columns:
#                 hits = int(sample[c].astype(str).str.contains(r"^https?://", na=False).sum())
#                 if hits > best_hits:
#                     best_col, best_hits = c, hits
#             if best_hits > 0:
#                 url_col = best_col

#         if url_col is None:
#             print("Error: Could not infer URL column.", file=sys.stderr)
#             sys.exit(2)

#         # Valid URLs
#         urls = chunk[url_col].astype(str).str.strip()
#         valid_mask = urls.notna() & (urls != "") & urls.str.contains(r"^https?://", na=False)
#         if not valid_mask.any():
#             continue

#         total_urls += int(valid_mask.sum())
#         sub = chunk.loc[valid_mask]

#         # Extract dates
#         dates = extract_dates_series(sub, path_col)
#         good = dates.notna()
#         unknown_total += int((~good).sum())

#         if good.any():
#             for d, n in dates[good].value_counts().items():
#                 per_date[d] += int(n)

#     # ===== Output =====
#     print("=== Counts per date (Startpage) ===")
#     if not per_date:
#         print("(No valid dates found.)")
#     else:
#         total_counted = sum(per_date.values())
#         for d in sorted(per_date):
#             cnt = per_date[d]
#             pct = (cnt / total_urls * 100.0) if total_urls else 0.0
#             print(f"{d} : {cnt}  ({pct:.2f}%)")

#         print(f"\nTotal URLs (all): {total_urls}")
#         print(f"Counted in dates: {total_counted}")
#         if unknown_total:
#             print(f"(Info) Rows with no valid date: {unknown_total}")

#     # Save CSV
#     if per_date:
#         out_rows = [{"date": d, "count": per_date[d]} for d in sorted(per_date)]
#         out_df = pd.DataFrame(out_rows, columns=["date", "count"])
#         if unknown_total:
#             out_df = pd.concat(
#                 [out_df, pd.DataFrame([{"date": "unknown", "count": unknown_total}])],
#                 ignore_index=True
#             )
#         out_df.to_csv(OUT_CSV, index=False)
#         print(f"\nSaved per-date counts to: {OUT_CSV}")


# if __name__ == "__main__":
#     main()
