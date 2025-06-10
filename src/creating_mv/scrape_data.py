import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def download_videos(base_url: str, output_dir: str, limit: int = 10):
    """Download video files from a web page.

    Parameters
    ----------
    base_url : str
        Page containing <video> or <a> tags linking to video files.
    output_dir : str
        Directory where downloaded videos will be stored.
    limit : int, optional
        Maximum number of videos to download, by default 10.
    """
    os.makedirs(output_dir, exist_ok=True)
    resp = requests.get(base_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    count = 0
    for tag in soup.find_all(["video", "a"]):
        src = tag.get("src") or tag.get("href")
        if not src:
            continue
        if not src.lower().endswith((".mp4", ".mov", ".webm")):
            continue
        video_url = urljoin(base_url, src)
        fname = os.path.join(output_dir, os.path.basename(src))
        with requests.get(video_url, stream=True) as r:
            r.raise_for_status()
            with open(fname, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        count += 1
        if count >= limit:
            break
    print(f"Downloaded {count} videos to {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download sample videos for training")
    parser.add_argument("url", nargs="?", help="URL of the page to scrape")
    parser.add_argument("--urls-file", help="File containing one URL per line")
    parser.add_argument("--out", default="data/raw", help="Output directory")
    parser.add_argument("--limit", type=int, default=10, help="Number of videos to download from each page")
    args = parser.parse_args()

    if not args.url and not args.urls_file:
        parser.error("provide a URL or --urls-file")

    urls = []
    if args.urls_file:
        with open(args.urls_file) as f:
            urls.extend([line.strip() for line in f if line.strip()])
    if args.url:
        urls.append(args.url)

    for idx, u in enumerate(urls, start=1):
        out_dir = os.path.join(args.out, f"url_{idx}") if len(urls) > 1 else args.out
        download_videos(u, out_dir, args.limit)


if __name__ == "__main__":
    main()
