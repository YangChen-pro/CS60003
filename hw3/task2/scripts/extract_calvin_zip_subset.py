"""Extract a small CALVIN subset from the official remote ZIP with HTTP ranges.

The official CALVIN server only exposes large ZIP files such as
`task_ABC_D.zip`. This script avoids downloading the full archive by:

1. reading the ZIP64 end records from the tail of the remote file;
2. caching the central directory locally;
3. selecting episode files by scene ranges recorded in `scene_info.npy`;
4. downloading only the selected compressed members and inflating them locally.

It uses only the Python standard library so it can run on the 135 server before
LeRobot/CALVIN dependencies are installed.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for bare Python environments.
    tqdm = None


DEFAULT_URL = "http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip"
ZIP64_EOCD_LOCATOR = b"PK\x06\x07"
ZIP64_EOCD = b"PK\x06\x06"
EOCD = b"PK\x05\x06"
CENTRAL_DIR_FILE = b"PK\x01\x02"
LOCAL_FILE = b"PK\x03\x04"


@dataclass(frozen=True)
class ZipEntry:
    """Metadata needed to fetch and extract one remote ZIP member."""

    name: str
    method: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int
    crc32: int


@dataclass(frozen=True)
class RemoteZipInfo:
    """ZIP64 archive offsets."""

    size: int
    central_dir_offset: int
    central_dir_size: int
    total_entries: int


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="Remote CALVIN ZIP URL.")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Directory for central-directory cache.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write selected files.")
    parser.add_argument("--split", choices=["training", "validation"], default="training")
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Scene names to sample, e.g. calvin_scene_A. Repeat for multiple scenes.",
    )
    parser.add_argument("--episodes-per-scene", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=8, help="Parallel member downloads.")
    parser.add_argument(
        "--include-lang-annotations",
        action="store_true",
        help="Also extract language annotation arrays. They are large and not needed for an episode-range probe.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only list selected files.")
    parser.add_argument("--list-selected", action="store_true", help="Print every selected ZIP member to stdout.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing extracted files.")
    return parser.parse_args()


def main() -> None:
    """Run subset extraction."""
    args = parse_args()
    scenes = args.scene or ["calvin_scene_A", "calvin_scene_B", "calvin_scene_C"]
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    zip_info = inspect_remote_zip(args.url)
    entries = load_entries(args.url, zip_info, args.cache_dir)
    entry_map = {entry.name: entry for entry in entries}
    root_prefix = common_root_prefix(entry_map)
    print(
        json.dumps(
            {
                "remote_size": zip_info.size,
                "central_dir_size": zip_info.central_dir_size,
                "entries": len(entries),
                "root_prefix": root_prefix,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    selected = select_subset(
        args.url,
        entry_map,
        root_prefix,
        args.split,
        scenes,
        args.episodes_per_scene,
        include_lang_annotations=args.include_lang_annotations,
    )
    selection_mode = "scene_info" if f"{root_prefix}{args.split}/scene_info.npy" in entry_map else "sorted_episode_ids"
    manifest = {
        "url": args.url,
        "split": args.split,
        "scenes": scenes,
        "selection_mode": selection_mode,
        "episodes_per_scene": args.episodes_per_scene,
        "include_lang_annotations": args.include_lang_annotations,
        "selected_count": len(selected),
        "compressed_bytes": sum(entry.compressed_size for entry in selected),
        "uncompressed_bytes": sum(entry.uncompressed_size for entry in selected),
        "selected": [entry.name for entry in selected],
    }
    print_manifest(manifest, args.list_selected)
    if args.dry_run:
        return

    extract_entries(args.url, selected, args.output_dir, args.force, args.jobs)
    write_json(args.output_dir / "subset_manifest.json", manifest)


def inspect_remote_zip(url: str) -> RemoteZipInfo:
    """Read ZIP64 end records from the remote file tail."""
    size = int(head_content_length(url))
    tail_start = max(0, size - 1024 * 1024)
    tail = http_range(url, tail_start, size - 1)
    eocd_pos = tail.rfind(EOCD)
    if eocd_pos < 0:
        raise ValueError("Unable to locate ZIP EOCD in remote tail.")
    locator_pos = tail.rfind(ZIP64_EOCD_LOCATOR, 0, eocd_pos)
    if locator_pos < 0:
        raise ValueError("Expected ZIP64 locator for CALVIN archives.")
    _, _, zip64_eocd_offset, _ = struct.unpack("<4sLQL", tail[locator_pos : locator_pos + 20])
    zip64_eocd = http_range(url, zip64_eocd_offset, zip64_eocd_offset + 55)
    if zip64_eocd[:4] != ZIP64_EOCD:
        raise ValueError("Invalid ZIP64 EOCD signature.")
    fields = struct.unpack("<4sQ2H2L4Q", zip64_eocd[:56])
    return RemoteZipInfo(
        size=size,
        total_entries=int(fields[7]),
        central_dir_size=int(fields[8]),
        central_dir_offset=int(fields[9]),
    )


def load_entries(url: str, info: RemoteZipInfo, cache_dir: Path) -> list[ZipEntry]:
    """Load or download the central directory, then parse entries."""
    cache_path = cache_dir / "task_ABC_D.central_directory.bin"
    meta_path = cache_dir / "task_ABC_D.central_directory.json"
    if not cache_path.exists() or cache_path.stat().st_size != info.central_dir_size:
        download_range_with_curl(url, info.central_dir_offset, info.central_dir_offset + info.central_dir_size - 1, cache_path)
        write_json(
            meta_path,
            {
                "url": url,
                "central_dir_offset": info.central_dir_offset,
                "central_dir_size": info.central_dir_size,
                "total_entries": info.total_entries,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
    return parse_central_directory(cache_path.read_bytes())


def parse_central_directory(data: bytes) -> list[ZipEntry]:
    """Parse central-directory bytes into ZipEntry objects."""
    entries: list[ZipEntry] = []
    pos = 0
    while pos + 46 <= len(data):
        if data[pos : pos + 4] != CENTRAL_DIR_FILE:
            raise ValueError(f"Bad central directory signature at offset {pos}")
        fields = struct.unpack("<4s6H3L5H2L", data[pos : pos + 46])
        method = fields[4]
        crc32 = fields[7]
        compressed_size = fields[8]
        uncompressed_size = fields[9]
        filename_length = fields[10]
        extra_length = fields[11]
        comment_length = fields[12]
        local_header_offset = fields[16]
        name_start = pos + 46
        extra_start = name_start + filename_length
        comment_start = extra_start + extra_length
        name = data[name_start:extra_start].decode("utf-8", errors="replace")
        extra = data[extra_start:comment_start]
        if 0xFFFFFFFF in {compressed_size, uncompressed_size, local_header_offset}:
            uncompressed_size, compressed_size, local_header_offset = apply_zip64_extra(
                extra,
                uncompressed_size,
                compressed_size,
                local_header_offset,
            )
        entries.append(
            ZipEntry(
                name=name,
                method=method,
                compressed_size=int(compressed_size),
                uncompressed_size=int(uncompressed_size),
                local_header_offset=int(local_header_offset),
                crc32=int(crc32),
            )
        )
        pos = comment_start + comment_length
    return entries


def apply_zip64_extra(extra: bytes, uncompressed: int, compressed: int, offset: int) -> tuple[int, int, int]:
    """Read ZIP64 sizes/offset from a central-directory extra field."""
    pos = 0
    values = [uncompressed, compressed, offset]
    while pos + 4 <= len(extra):
        header_id, data_size = struct.unpack("<HH", extra[pos : pos + 4])
        payload = extra[pos + 4 : pos + 4 + data_size]
        if header_id == 0x0001:
            cursor = 0
            for idx, current in enumerate(values):
                if current == 0xFFFFFFFF:
                    values[idx] = struct.unpack("<Q", payload[cursor : cursor + 8])[0]
                    cursor += 8
            return int(values[0]), int(values[1]), int(values[2])
        pos += 4 + data_size
    raise ValueError("Missing ZIP64 extra data for large entry.")


def common_root_prefix(entry_map: dict[str, ZipEntry]) -> str:
    """Return the top-level archive prefix, usually `task_ABC_D/`."""
    first = next(name for name in entry_map if "/" in name)
    return first.split("/", 1)[0] + "/"


def select_subset(
    url: str,
    entry_map: dict[str, ZipEntry],
    root_prefix: str,
    split: str,
    scenes: Iterable[str],
    episodes_per_scene: int,
    include_lang_annotations: bool,
) -> list[ZipEntry]:
    """Select metadata and the first N episode frames for each requested scene."""
    scene_info_name = f"{root_prefix}{split}/scene_info.npy"
    selected_names = metadata_names(root_prefix, split, include_lang_annotations)
    selected_names = [name for name in selected_names if name in entry_map]
    if scene_info_name not in entry_map:
        selected_names.extend(select_sorted_episode_names(entry_map, root_prefix, split, episodes_per_scene))
        return entries_from_names(entry_map, selected_names)

    scene_info_bytes = read_entry(url, entry_map[scene_info_name])
    scene_info = load_npy_dict(scene_info_bytes)
    for scene in scenes:
        if scene not in scene_info:
            raise KeyError(f"Scene {scene!r} not found in {scene_info_name}; available={list(scene_info)}")
        start, end = [int(value) for value in scene_info[scene]]
        for episode_id in range(start, min(end, start + episodes_per_scene - 1) + 1):
            selected_names.append(f"{root_prefix}{split}/episode_{episode_id:07d}.npz")
    return entries_from_names(entry_map, selected_names)


def select_sorted_episode_names(
    entry_map: dict[str, ZipEntry],
    root_prefix: str,
    split: str,
    limit: int,
) -> list[str]:
    """Select the first N episode files when a split has no scene_info.npy."""
    prefix = f"{root_prefix}{split}/episode_"
    names = [name for name in entry_map if name.startswith(prefix) and name.endswith(".npz")]
    return sorted(names)[:limit]


def entries_from_names(entry_map: dict[str, ZipEntry], names: list[str]) -> list[ZipEntry]:
    """Resolve selected member names and preserve order without duplicates."""
    missing = [name for name in names if name not in entry_map]
    if missing:
        raise FileNotFoundError(f"Selected entries missing from central directory: {missing[:10]}")
    return [entry_map[name] for name in dict.fromkeys(names)]


def metadata_names(root_prefix: str, split: str, include_lang_annotations: bool) -> list[str]:
    """Return split metadata files useful for a minimal CALVIN subset."""
    base = f"{root_prefix}{split}"
    names = [
        f"{base}/.hydra/config.yaml",
        f"{base}/.hydra/hydra.yaml",
        f"{base}/.hydra/merged_config.yaml",
        f"{base}/.hydra/overrides.yaml",
        f"{base}/ep_lens.npy",
        f"{base}/ep_start_end_ids.npy",
        f"{base}/scene_info.npy",
        f"{base}/statistics.yaml",
    ]
    if include_lang_annotations:
        names.extend(
            [
                f"{base}/lang_annotations/auto_lang_ann.npy",
                f"{base}/lang_annotations/embeddings.npy",
                f"{base}/lang_paraphrase-MiniLM-L3-v2/auto_lang_ann.npy",
                f"{base}/lang_paraphrase-MiniLM-L3-v2/embeddings.npy",
            ]
        )
    return names


def extract_entry(url: str, entry: ZipEntry, output_dir: Path, force: bool) -> int:
    """Fetch, decompress, and write one entry under output_dir."""
    target = output_dir / entry.name
    if target.exists() and not force:
        return entry.compressed_size
    if entry.name.endswith("/"):
        target.mkdir(parents=True, exist_ok=True)
        return 0
    data = read_entry(url, entry)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    if len(data) != entry.uncompressed_size:
        raise IOError(f"Size mismatch for {entry.name}: {len(data)} != {entry.uncompressed_size}")
    actual_crc = zlib.crc32(data) & 0xFFFFFFFF
    if actual_crc != entry.crc32:
        raise IOError(f"CRC mismatch for {entry.name}: {actual_crc:x} != {entry.crc32:x}")
    return entry.compressed_size


def extract_entries(url: str, entries: list[ZipEntry], output_dir: Path, force: bool, jobs: int) -> None:
    """Extract selected members with a tqdm byte progress bar."""
    total = sum(entry.compressed_size for entry in entries)
    progress = make_progress(total=total, desc="extract selected members", unit="B", unit_scale=True)
    workers = max(1, jobs)
    try:
        if workers == 1:
            for entry in entries:
                progress.update(extract_entry(url, entry, output_dir, force))
            return
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(extract_entry, url, entry, output_dir, force) for entry in entries]
            for future in as_completed(futures):
                progress.update(future.result())
    finally:
        progress.close()


def read_entry(url: str, entry: ZipEntry) -> bytes:
    """Read and inflate one remote ZIP member."""
    header = http_range(url, entry.local_header_offset, entry.local_header_offset + 29)
    if header[:4] != LOCAL_FILE:
        raise ValueError(f"Bad local header for {entry.name}")
    fields = struct.unpack("<4s5H3L2H", header)
    filename_length = fields[9]
    extra_length = fields[10]
    data_start = entry.local_header_offset + 30 + filename_length + extra_length
    compressed = http_range(url, data_start, data_start + entry.compressed_size - 1)
    if entry.method == 0:
        return compressed
    if entry.method == 8:
        return zlib.decompress(compressed, -zlib.MAX_WBITS)
    raise NotImplementedError(f"Unsupported ZIP compression method {entry.method} for {entry.name}")


def load_npy_dict(data: bytes) -> dict:
    """Load a npy dictionary without importing project dependencies."""
    import numpy as np

    value = np.load(io.BytesIO(data), allow_pickle=True)
    if value.shape != ():
        raise ValueError("Expected scalar object npy for scene_info.")
    obj = value.item()
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in scene_info, got {type(obj).__name__}")
    return obj


def head_content_length(url: str) -> str:
    """Return Content-Length via HEAD."""
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.headers["Content-Length"]


def http_range(url: str, start: int, end: int) -> bytes:
    """Fetch inclusive byte range from a URL."""
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def download_range_with_curl(url: str, start: int, end: int, output: Path) -> None:
    """Download a large byte range using curl for better resume/progress behavior."""
    output.parent.mkdir(parents=True, exist_ok=True)
    curl = shutil.which("curl")
    if not curl:
        output.write_bytes(http_range(url, start, end))
        return
    byte_count = end - start + 1
    if byte_count > 32 * 1024 * 1024:
        download_range_chunks_with_curl(url, start, end, output, curl)
        return
    tmp = output.with_suffix(output.suffix + ".tmp")
    subprocess.run(
        [
            curl,
            "-L",
            "--fail",
            "--retry",
            "5",
            "--range",
            f"{start}-{end}",
            "--output",
            str(tmp),
            url,
        ],
        check=True,
    )
    tmp.replace(output)


def download_range_chunks_with_curl(url: str, start: int, end: int, output: Path, curl: str) -> None:
    """Download a large range as independent chunks, then concatenate them."""
    chunk_size = int(os.environ.get("CALVIN_RANGE_CHUNK_SIZE", str(8 * 1024 * 1024)))
    jobs = int(os.environ.get("CALVIN_RANGE_JOBS", "8"))
    chunk_dir = output.with_suffix(output.suffix + ".chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    ranges: list[tuple[int, int, Path]] = []
    chunk_start = start
    index = 0
    while chunk_start <= end:
        chunk_end = min(end, chunk_start + chunk_size - 1)
        ranges.append((chunk_start, chunk_end, chunk_dir / f"chunk_{index:05d}.bin"))
        chunk_start = chunk_end + 1
        index += 1

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(download_one_chunk, curl, url, chunk_start, chunk_end, path) for chunk_start, chunk_end, path in ranges]
        progress = make_progress(total=end - start + 1, desc="download central directory", unit="B", unit_scale=True)
        try:
            for future in as_completed(futures):
                progress.update(future.result())
        finally:
            progress.close()

    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("wb") as handle:
        for _, _, path in ranges:
            handle.write(path.read_bytes())
    expected_size = end - start + 1
    actual_size = tmp.stat().st_size
    if actual_size != expected_size:
        raise IOError(f"Central directory cache size mismatch: {actual_size} != {expected_size}")
    tmp.replace(output)
    shutil.rmtree(chunk_dir, ignore_errors=True)


def download_one_chunk(curl: str, url: str, start: int, end: int, output: Path) -> int:
    """Download one byte range if absent or incomplete."""
    expected_size = end - start + 1
    if output.exists() and output.stat().st_size == expected_size:
        return expected_size
    tmp = output.with_suffix(output.suffix + ".tmp")
    subprocess.run(
        [
            curl,
            "-sS",
            "-L",
            "--fail",
            "--retry",
            "5",
            "--range",
            f"{start}-{end}",
            "--output",
            str(tmp),
            url,
        ],
        check=True,
    )
    actual_size = tmp.stat().st_size
    if actual_size != expected_size:
        raise IOError(f"Chunk size mismatch for {start}-{end}: {actual_size} != {expected_size}")
    tmp.replace(output)
    return expected_size


def make_progress(**kwargs):
    """Create a tqdm progress bar, or a no-op progress object if tqdm is absent."""
    if tqdm is not None:
        return tqdm(**kwargs)
    return NoProgress()


class NoProgress:
    """Minimal tqdm-compatible fallback."""

    def update(self, _: int) -> None:
        return

    def close(self) -> None:
        return


def write_json(path: Path, data: dict) -> None:
    """Write JSON with stable UTF-8 formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def print_manifest(manifest: dict, list_selected: bool) -> None:
    """Print a compact manifest by default so tqdm remains readable."""
    if list_selected:
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
        return
    compact = dict(manifest)
    selected = compact.pop("selected", [])
    compact["selected_preview"] = selected[:5]
    compact["selected_omitted"] = max(0, len(selected) - 5)
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
