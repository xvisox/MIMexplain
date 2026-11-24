"""Download a random subset of the Pexels-110k-512p dataset with labels.

Thin data-fetch utility for the Grad-ECLIP experiments. It does NOT pull the
full ~29.4 GB archive and does NOT use the rate-limited file-list API. Instead
it exploits two facts about Kaggle's dataset download:

  * the download endpoint 302-redirects to a Google Cloud Storage URL that
    supports HTTP Range requests, and
  * a ZIP's central directory (the index of every entry) lives at the *end* of
    the file.

So we open the remote ZIP through a Range-backed file object, let ``zipfile``
read just the central directory (a few MB -> the full list of ~110k filenames,
in ~4 requests), randomly sample N entries, and Range-read only those entries'
bytes. Bandwidth is the index plus the N sampled images -- nothing else -- and
the sample is uniformly random across the whole dataset.

This also sidesteps the slug-vs-id filename problem entirely: the central
directory gives the exact stored filename for every image, and img_id is parsed
from it (trailing digits before the extension).

Outputs (under an ignored ``data/`` dir by default):
    <out>/images/<img_id>.jpg   the sampled images (renamed to bare img_id)
    <out>/labels.csv            metadata rows for exactly the sampled images
    <out>/labels.parquet        same, if a parquet engine is available

Usage (deps are declared in pyproject; plain ``uv run`` works):
    uv run scripts/download_pexels_subset.py --validate-only   # auth + index + meta
    uv run scripts/download_pexels_subset.py --num 1000        # real download

Kaggle auth (any of):
    * `kaggle auth login` (OAuth; writes ~/.kaggle/credentials.json)
    * ~/.kaggle/kaggle.json  (classic token, chmod 600)
    * export KAGGLE_USERNAME=... KAGGLE_KEY=...
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path

DATASET = "innominate817/pexels-110k-512p-min-jpg"
DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}"
METADATA_FILE = "pexels-110k-512p-min-jpg/attributes_df.json"
IMAGE_DIR_PREFIX = "pexels-110k-512p-min-jpg/images/"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
READ_AHEAD = 1 << 20  # 1 MiB: covers a whole image entry in one range request
# Trailing run of digits immediately before the file extension == img_id.
_ID_RE = re.compile(r"(\d+)(?=\.[^.]+$)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num", type=int, default=1000, help="number of images to sample")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/pexels-1k"),
        help="output directory (kept local; data/ is gitignored)",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="auth + read ZIP index + load metadata, then exit before downloading",
    )
    return p.parse_args()


def refresh_oauth_token() -> None:
    """Use the kaggle singleton to refresh an expired OAuth token on disk.

    kagglehub and our raw HTTP calls both read ~/.kaggle/credentials.json; the
    singleton is the only thing that refreshes it. Harmless for classic-key auth
    (no-op). ``import kaggle`` hard-exits at import only when creds are absent.
    """
    try:
        from kaggle import api

        api.authenticate()
        api.dataset_list_files(DATASET, page_size=1)  # forces token refresh
        print("[auth] kaggle credentials OK (token refreshed)")
    except SystemExit as exc:
        print(
            "[auth] not authenticated. Run `kaggle auth login`, or set "
            "~/.kaggle/kaggle.json / KAGGLE_USERNAME+KAGGLE_KEY, then re-run.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    except Exception as exc:  # noqa: BLE001
        print(f"[auth] authentication failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _kaggle_request_kwargs() -> dict:
    """Auth kwargs for a requests call to the Kaggle download endpoint."""
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if user and key:
        return {"auth": (user, key)}
    kjson = Path.home() / ".kaggle" / "kaggle.json"
    if kjson.exists():
        d = json.loads(kjson.read_text())
        return {"auth": (d["username"], d["key"])}
    creds = Path.home() / ".kaggle" / "credentials.json"
    if creds.exists():
        tok = json.loads(creds.read_text())["access_token"]
        return {"headers": {"Authorization": f"Bearer {tok}"}}
    raise SystemExit("[auth] no Kaggle credentials found")


def resolve_archive_target() -> tuple[str, dict]:
    """Resolve the archive to a (url, request_kwargs) pair for Range GETs.

    Follows the Kaggle download redirect to the signed GCS URL (which needs no
    auth and supports ranges). If the endpoint serves bytes directly instead of
    redirecting, return it with auth kwargs.
    """
    import requests

    a = _kaggle_request_kwargs()
    r = requests.get(DOWNLOAD_URL, allow_redirects=False, timeout=60, **a)
    if r.is_redirect or r.status_code in (301, 302, 303, 307, 308):
        return r.headers["location"], {}
    r.raise_for_status()
    return DOWNLOAD_URL, a


class HttpRangeFile(io.RawIOBase):
    """A seekable, read-only file backed by HTTP Range requests.

    Re-resolves the (possibly expiring, signed) URL on a 403. A 1 MiB read-ahead
    buffer means reading a whole image entry costs a single range request.
    """

    def __init__(self) -> None:
        import requests

        self._requests = requests
        self.url, self.kwargs = resolve_archive_target()
        self.size = int(self._requests.head(self.url, timeout=60, **self.kwargs)
                         .headers["content-length"])
        self.pos = 0
        self.reqs = 0
        self._buf = b""
        self._buf_start = 0  # file offset the buffer begins at

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:  # SEEK_END
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def _http_range(self, start: int, end_inclusive: int) -> bytes:
        for _ in range(4):
            r = self._requests.get(
                self.url,
                headers={**self.kwargs.get("headers", {}),
                         "Range": f"bytes={start}-{end_inclusive}"},
                auth=self.kwargs.get("auth"),
                timeout=120,
            )
            if r.status_code == 403:  # signed URL expired -> re-resolve
                self.url, self.kwargs = resolve_archive_target()
                continue
            r.raise_for_status()
            self.reqs += 1
            return r.content
        raise OSError("repeated 403 fetching range; auth likely expired")

    def read(self, n: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        if n is None or n < 0:  # read to EOF (rare for zipfile)
            data = self._http_range(self.pos, self.size - 1)
            self.pos += len(data)
            return data
        end = min(self.pos + n, self.size)
        if not (self._buf_start <= self.pos and end <= self._buf_start + len(self._buf)):
            block_end = min(self.pos + max(n, READ_AHEAD), self.size) - 1
            self._buf = self._http_range(self.pos, block_end)
            self._buf_start = self.pos
        rel = self.pos - self._buf_start
        data = self._buf[rel:rel + (end - self.pos)]
        self.pos += len(data)
        return data


def parse_img_id(name: str) -> str | None:
    m = _ID_RE.search(Path(name).name)
    return m.group(1) if m else None


def load_metadata():
    """Load the attributes table; img_id is the JSON index -> promote to column."""
    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    print(f"[meta] loading '{METADATA_FILE}' ...")
    df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, DATASET, METADATA_FILE)
    df = df.rename_axis("img_id").reset_index()
    df["img_id"] = df["img_id"].astype(str)
    print(f"[meta] loaded {len(df):,} rows; columns: {list(df.columns)}")
    return df


def main() -> None:
    args = parse_args()
    refresh_oauth_token()

    import pandas as pd

    rf = HttpRangeFile()
    print(f"[zip] archive ~{rf.size / 1e9:.1f} GB; reading central directory ...")
    zf = zipfile.ZipFile(rf)
    image_names = [
        n for n in zf.namelist()
        if n.startswith(IMAGE_DIR_PREFIX) and n.lower().endswith(IMAGE_EXTS)
    ]
    print(f"[zip] {len(image_names):,} image entries indexed in {rf.reqs} range requests")

    index = pd.DataFrame(
        [(parse_img_id(n), n) for n in image_names], columns=["img_id", "name"]
    ).dropna()

    meta = load_metadata()
    usable = index.merge(meta[["img_id"]], on="img_id", how="inner")
    print(f"[sample] {len(usable):,} images have both a ZIP entry and metadata")

    if args.validate_only:
        print("[validate-only] auth + ZIP index + metadata OK; exiting before download.")
        return

    n = min(args.num, len(usable))
    sample = usable.sample(n=n, random_state=args.seed).reset_index(drop=True)
    print(f"[sample] selected {n} images uniformly (seed={args.seed})")

    img_dir = args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    ok, skipped, failed = 0, 0, []
    for i, row in enumerate(sample.itertuples(index=False), 1):
        dest = img_dir / f"{row.img_id}.jpg"
        if dest.exists():
            skipped += 1
        else:
            try:
                with zf.open(row.name) as src, open(dest, "wb") as out:
                    out.write(src.read())
                ok += 1
            except Exception as exc:  # noqa: BLE001
                failed.append((row.img_id, str(exc)))
        if i % 50 == 0 or i == n:
            print(
                f"[download] {i}/{n} (ok={ok} skipped={skipped} "
                f"failed={len(failed)} range_reqs={rf.reqs})"
            )

    labels = sample.rename(columns={"name": "path"}).merge(meta, on="img_id", how="left")
    args.out.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.out / "labels.csv", index=False)
    try:
        labels.to_parquet(args.out / "labels.parquet", index=False)
    except Exception:  # noqa: BLE001
        print("[labels] parquet skipped (no pyarrow/fastparquet); CSV written.")

    print(
        f"\n[done] images: ok={ok} skipped(existing)={skipped} failed={len(failed)}\n"
        f"       images dir : {img_dir}\n"
        f"       labels     : {args.out / 'labels.csv'}"
    )
    if failed:
        print(f"[warn] {len(failed)} downloads failed; first few: {failed[:5]}")


if __name__ == "__main__":
    main()
