import argparse
import os
from typing import Iterable, Optional

import requests
import cv2
from tqdm import tqdm

def download_file(
    url: str,
    local_filename: str,
    timeout: int = 30,
    max_bytes: Optional[int] = None,
) -> bool:
    """Download a file from a URL with a progress bar.

    Returns True on success, False otherwise.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(local_filename), exist_ok=True)

    print(f"Downloading from: {url}")
    try:
        headers = {
            # Helps avoid some host blocks and ensures binary download for releases
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/octet-stream, */*"
        }
        with requests.get(url, stream=True, headers=headers, timeout=timeout) as r:
            r.raise_for_status()
            total = r.headers.get("content-length")
            try:
                total = int(total) if total else None
            except ValueError:
                total = None

            # Enforce size cap if provided and server returns content-length
            if max_bytes is not None and total is not None and total > max_bytes:
                print(
                    f"Skipping download: file size {total/1_048_576:.1f} MiB exceeds limit "
                    f"{max_bytes/1_048_576:.1f} MiB for URL {url}"
                )
                return False

            block_size = 1024 * 64  # 64 KiB chunks for decent throughput
            progress_bar = tqdm(
                total=total,
                unit="iB",
                unit_scale=True,
                desc=os.path.basename(local_filename) or "download",
            )
            with open(local_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    progress_bar.update(len(chunk))
            progress_bar.close()

            # If server provided total length, validate we received all bytes
            if total is not None and progress_bar.n != total:
                print("ERROR: Download incomplete (size mismatch).")
                return False

        # --- Post-download validation ---
        print("Validating downloaded file as a video...")
        cap = None
        try:
            cap = cv2.VideoCapture(local_filename)
            if not cap.isOpened():
                raise RuntimeError("File could not be opened by OpenCV.")
            is_read, _ = cap.read()
            if not is_read:
                raise RuntimeError("First frame could not be read.")
            print("Validation successful.")
        except Exception as e:
            print(f"ERROR: Downloaded file is not a valid video. {e}")
            os.remove(local_filename)
            print(f"Removed invalid file: {local_filename}")
            return False
        finally:
            if cap:
                cap.release()

        print(f"Successfully downloaded to: {local_filename}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during download: {e}")
        return False


def download_from_mirrors(urls: Iterable[str], out_path: str, max_bytes: Optional[int] = None) -> bool:
    """Try downloading from the first working URL in the provided list."""
    for url in urls:
        if download_file(url, out_path, max_bytes=max_bytes):
            return True
        print(f"Retrying with next mirror... (failed: {url})")
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a sample video for testing.")
    parser.add_argument(
        "--url",
        default=None,
        help="Direct URL to download. If omitted, tries known mirrors.",
    )
    parser.add_argument(
        "--out",
        default="sample_data/sample_video.mp4",
        help="Output path for the downloaded file.",
    )
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=25.0,
        help="Maximum allowed download size in MiB (based on Content-Length). Set to 0 to disable.",
    )
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="Include large fallback mirrors (e.g., Big Buck Bunny ~158MB).",
    )
    args = parser.parse_args()

    # A list of mirror URLs. The first one is small and ideal for quick testing.
    # The script will try them in order until one succeeds.
    mirrors = [
        # Small, short video for quick development and testing
        "https://filesamples.com/samples/video/mp4/sample_640x360.mp4",
        # Ultralytics bus (may 404 depending on release availability)
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.mp4",
    ]

    # Optionally include a large public demo video as last resort
    if args.include_large:
        mirrors.append("https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4")

    urls = [args.url] if args.url else mirrors
    max_bytes = None if args.max_size_mb <= 0 else int(args.max_size_mb * 1024 * 1024)
    success = download_from_mirrors(urls, args.out, max_bytes=max_bytes)
    if not success:
        raise SystemExit(1)
