import os
from concurrent.futures import ThreadPoolExecutor

from .scrape_data import download_videos


def _scrape_job(args):
    idx, url, base_out, limit = args
    out_dir = os.path.join(base_out, f"url_{idx}")
    download_videos(url, out_dir, limit)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download videos from multiple URLs concurrently")
    parser.add_argument("urls_file", help="File containing one URL per line")
    parser.add_argument("--out", default="data/raw", help="Base output directory")
    parser.add_argument("--limit", type=int, default=10, help="Max videos per URL")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    with open(args.urls_file) as f:
        urls = [line.strip() for line in f if line.strip()]

    os.makedirs(args.out, exist_ok=True)

    jobs = [(i, u, args.out, args.limit) for i, u in enumerate(urls, start=1)]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for _ in ex.map(_scrape_job, jobs):
            pass


if __name__ == "__main__":
    main()
