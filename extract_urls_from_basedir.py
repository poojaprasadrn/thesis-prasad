import os
import tarfile
import gzip
import re
from pathlib import PurePosixPath
from warcio.archiveiterator import ArchiveIterator
from bs4 import BeautifulSoup
import pandas as pd

# --- Constants ---
WARC_BASE_DIR = "/mnt/ceph/storage/data-in-progress/data-research/web-search/affiliate-serp-crawls"
OUTPUT_CSV = "warc_tar_extracted_urls.csv"

# --- HTML helpers ---
def extract_text_from_html(html):
    try:
        soup = BeautifulSoup(html, "html.parser")
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        text = soup.get_text(separator=' ')
        return ' '.join(text.split())
    except Exception as e:
        print(f"❌ HTML parse failed: {e}")
        return ""

def extract_url_from_html(html):
    try:
        soup = BeautifulSoup(html, "html.parser")

        # og:url
        og_url = soup.find("meta", property="og:url")
        if og_url and og_url.get("content"):
            url = og_url.get("content").strip()
            if url.startswith("//"):
                url = "https:" + url
            print("🔍 URL found via og:url")
            return url

        # canonical link
        canonical = soup.find("link", rel="canonical")
        if canonical and canonical.get("href"):
            url = canonical["href"].strip()
            if url.startswith("//"):
                url = "https:" + url
            print("🔍 URL found via canonical link")
            return url

        # base href
        base_tag = soup.find("base", href=True)
        if base_tag:
            url = base_tag["href"].strip()
            if url.startswith("//"):
                url = "https:" + url
            print("🔍 URL found via base href")
            return url

    except Exception as e:
        print(f"⚠️ Structured URL extraction failed: {e}")

    # Fallback to regex
    match = re.search(r'<meta[^>]*property=["\']og:url["\'][^>]*content=["\']([^"\']+)["\']', html, re.IGNORECASE)
    if match:
        url = match.group(1).strip()
        if url.startswith("//"):
            url = "https:" + url
        print("🔍 URL found via regex og:url")
        return url

    match = re.search(r'<link[^>]*rel=["\']canonical["\'][^>]*href=["\']([^"\']+)["\']', html, re.IGNORECASE)
    if match:
        url = match.group(1).strip()
        if url.startswith("//"):
            url = "https:" + url
        print("🔍 URL found via regex canonical")
        return url

    match = re.search(r'<base[^>]*href=["\']([^"\']+)["\']', html, re.IGNORECASE)
    if match:
        url = match.group(1).strip()
        if url.startswith("//"):
            url = "https:" + url
        print("🔍 URL found via regex base")
        return url

    return "UNKNOWN_URL"

# --- Archive handlers ---
def open_archive(file_path):
    if file_path.endswith(".tar"):
        return tarfile.open(file_path, "r")
    elif file_path.endswith(".tar.xz"):
        return tarfile.open(file_path, "r:xz")
    elif file_path.endswith(".tar.gz") or file_path.endswith(".tgz"):
        return tarfile.open(file_path, "r:gz")
    else:
        raise ValueError(f"Unsupported tar format: {file_path}")

def process_tar(file_path):
    results = []
    try:
        tar = open_archive(file_path)
        members = tar.getnames()
        dom_paths = [m for m in members if m.endswith("dom.html")]

        for dom_path in dom_paths:
            query_path = "/".join(dom_path.split("/")[:-1])
            url_path = str(PurePosixPath(query_path) / "url")
            url = "UNKNOWN"

            try:
                if url_path in members:
                    url_file = tar.extractfile(url_path)
                    if url_file:
                        url = url_file.read().decode("utf-8").strip()

                html_file = tar.extractfile(dom_path)
                if html_file is None:
                    print(f"⚠️ Missing dom.html: {dom_path}")
                    continue

                html_bytes = html_file.read()
                html = html_bytes.decode("utf-8", errors="replace")

                # Fallback to HTML if URL is missing or invalid
                if not url or url.lower() in ["unknown", "unknown_url", "/", ""]:
                    url = extract_url_from_html(html)

                # Final validation
                if not url or url.lower() in ["unknown", "unknown_url", "/", ""] or not url.startswith("http"):
                    print("⚠️ Skipping: no valid URL found.")
                    continue

                text = extract_text_from_html(html)
                if len(text.split()) < 20:
                    continue

                results.append({
                    "URL": url,
                    "source_file": file_path,
                    "internal_path": dom_path,
                    "text": text
                })
                print(f"✅ TAR: {url} ({len(text.split())} words)")

            except Exception as e:
                print(f"⚠️ Error processing {dom_path}: {e}")
        tar.close()

    except Exception as e:
        print(f"❌ Cannot open TAR file {file_path}: {e}")
    return results

def process_warc(warc_path):
    results = []
    try:
        stream = gzip.open(warc_path, "rb") if warc_path.endswith(".gz") else open(warc_path, "rb")
        for record in ArchiveIterator(stream):
            if record.rec_type != "response":
                continue

            url = record.rec_headers.get_header("WARC-Target-URI")
            html_bytes = record.content_stream().read()
            html = html_bytes.decode("utf-8", errors="replace")
            text = extract_text_from_html(html)

            if not url or not url.startswith("http") or len(text.split()) < 20:
                continue

            results.append({
                "URL": url,
                "source_file": warc_path,
                "internal_path": "",
                "text": text
            })
            print(f"✅ WARC: {url} ({len(text.split())} words)")
        stream.close()

    except Exception as e:
        print(f"❌ Cannot open/process WARC file: {warc_path}: {e}")
    return results

# --- Main loop ---
def walk_and_process(base_dir):
    all_results = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"\n📦 Processing: {file_path}")

            if file.endswith(('.tar', '.tar.gz', '.tar.xz', '.tgz')):
                all_results.extend(process_tar(file_path))
            elif file.endswith(('.warc', '.warc.gz')):
                all_results.extend(process_warc(file_path))
            else:
                print(f"⚠️ Skipped unsupported file type: {file_path}")

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ All data saved to {OUTPUT_CSV}")

# --- Entry point ---
if __name__ == "__main__":
    walk_and_process(WARC_BASE_DIR)
