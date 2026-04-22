from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

try:
    import webdataset as wds
except ImportError:  # pragma: no cover - optional dependency in the current env
    wds = None

try:
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - optional dependency in the current env
    DataLoader = None


DEFAULT_RS5M_ROOT = Path("/media/disk5/dataset/RS5M")
DEFAULT_OUTPUT_ROOT = Path("/media/disk5/dataset/RS5M_offload")
DEFAULT_BATCH_SIZE = 128
DEFAULT_LOADER_WORKERS = 4
DEFAULT_WRITE_WORKERS = min(32, max(4, os.cpu_count() or 4))
DEFAULT_SUBDIR_SIZE = 1000
DEFAULT_INDEX_WIDTH = 7
REQUIRED_SAMPLE_KEYS = {"img_content", "img_name", "caption"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offload the RS5M webdataset into one png + one json file per sample."
        )
    )
    parser.add_argument(
        "--rs5m-root",
        type=Path,
        default=DEFAULT_RS5M_ROOT,
        help="Root directory that stores RS5M webdataset tar files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory used to store offloaded png/json files.",
    )
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=DEFAULT_LOADER_WORKERS,
        help="Number of DataLoader workers used by webdataset mode.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="How many samples are fetched in one webdataset batch.",
    )
    parser.add_argument(
        "--write-workers",
        type=int,
        default=DEFAULT_WRITE_WORKERS,
        help="Number of threads used to save png/json files concurrently.",
    )
    parser.add_argument(
        "--subdir-size",
        type=int,
        default=DEFAULT_SUBDIR_SIZE,
        help="How many images are stored in one subXXXX directory.",
    )
    parser.add_argument(
        "--index-width",
        type=int,
        default=DEFAULT_INDEX_WIDTH,
        help="Zero padding width for img_<index>.png/json filenames.",
    )
    parser.add_argument(
        "--max-pending-writes",
        type=int,
        default=None,
        help="Upper bound of in-flight write tasks.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for debugging or partial offload.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        help="Print progress every N completed samples.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Only print one sample preview and exit.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation after preview and continue directly.",
    )
    parser.add_argument(
        "--prefer-tar-fallback",
        action="store_true",
        help="Read tar files directly even if webdataset is installed.",
    )
    return parser.parse_args()


def byte_decode(value: bytes | str) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8").rstrip("\n")
    return value


def list_collate(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return samples


def build_train_url(train_dir: Path) -> str:
    return os.path.join(str(train_dir), "{pub11,rs3}-train-{0000..0031}.tar")


def my_decoder(key: str, value: Any) -> Any:
    if key.endswith(".img_content"):
        return value
    if key.endswith(".img_name") or key.endswith(".caption"):
        return byte_decode(value)
    return value


def get_wds_loader(train_dir: Path, num_workers: int, batch_size: int):
    if wds is None or DataLoader is None:
        raise RuntimeError(
            "webdataset/torch is not available in the current environment."
        )

    train_dataset = wds.WebDataset(
        build_train_url(train_dir),
        shardshuffle=False,
    ).decode(my_decoder)
    return DataLoader(
        train_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=list_collate,
        persistent_workers=num_workers > 0,
    )


def normalize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    if "img_content" not in sample:
        raise KeyError(f"Missing img_content in sample keys: {sorted(sample.keys())}")

    return {
        "sample_key": byte_decode(sample.get("__key__", "")),
        "source_url": byte_decode(sample.get("__url__", "")),
        "img_name": byte_decode(sample["img_name"]),
        "caption": byte_decode(sample["caption"]),
        "img_content": sample["img_content"],
    }


def run_wds_dataloader(train_dataloader) -> Iterator[dict[str, Any]]:
    for batch in train_dataloader:
        for sample in batch:
            yield normalize_sample(sample)


def iter_tar_paths(train_dir: Path) -> Iterator[Path]:
    for prefix in ("pub11", "rs3"):
        for tar_path in sorted(train_dir.glob(f"{prefix}-train-*.tar")):
            yield tar_path


def iter_tar_samples(tar_path: Path) -> Iterator[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}

    with tarfile.open(tar_path, "r|*") as tar_fp:
        for member in tar_fp:
            if not member.isfile():
                continue

            member_name = Path(member.name).name
            if "." not in member_name:
                continue

            sample_key, suffix = member_name.rsplit(".", 1)
            if suffix not in REQUIRED_SAMPLE_KEYS:
                continue

            extracted = tar_fp.extractfile(member)
            if extracted is None:
                continue

            value = extracted.read()
            group = groups.setdefault(
                sample_key,
                {
                    "__key__": sample_key,
                    "__url__": str(tar_path),
                },
            )
            group[suffix] = value if suffix == "img_content" else byte_decode(value)

            if REQUIRED_SAMPLE_KEYS.issubset(group):
                yield normalize_sample(group)
                del groups[sample_key]

    if groups:
        missing = {
            key: sorted(REQUIRED_SAMPLE_KEYS - set(sample.keys()))
            for key, sample in groups.items()
            if not REQUIRED_SAMPLE_KEYS.issubset(sample)
        }
        if missing:
            print(
                f"Warning: skipped {len(missing)} incomplete samples in {tar_path}",
                file=sys.stderr,
            )


def iter_samples(
    train_dir: Path,
    num_workers: int,
    batch_size: int,
    prefer_tar_fallback: bool = False,
) -> Iterator[dict[str, Any]]:
    use_webdataset = (
        not prefer_tar_fallback and wds is not None and DataLoader is not None
    )

    if use_webdataset:
        yield from run_wds_dataloader(get_wds_loader(train_dir, num_workers, batch_size))
        return

    if not prefer_tar_fallback:
        print(
            "webdataset or torch is unavailable, falling back to direct tar reading.",
            file=sys.stderr,
        )

    tar_paths = list(iter_tar_paths(train_dir))
    if not tar_paths:
        raise FileNotFoundError(f"No RS5M tar files found under: {train_dir}")

    for tar_path in tar_paths:
        yield from iter_tar_samples(tar_path)


def build_metadata(
    sample: dict[str, Any],
    output_root: Path,
    sample_index: int,
    subdir_size: int,
    index_width: int,
) -> dict[str, Any]:
    _, _, _, rel_png_path = resolve_output_paths(
        output_root=output_root,
        sample_index=sample_index,
        subdir_size=subdir_size,
        index_width=index_width,
    )
    return {
        "image_rel_path": rel_png_path.as_posix(),
        "caption": sample["caption"],
        "source_url": sample["source_url"],
    }


def print_one_sample_preview(
    train_dir: Path,
    output_root: Path,
    num_workers: int,
    batch_size: int,
    subdir_size: int,
    index_width: int,
    prefer_tar_fallback: bool = False,
) -> None:
    sample_iter = iter_samples(
        train_dir=train_dir,
        num_workers=num_workers,
        batch_size=batch_size,
        prefer_tar_fallback=prefer_tar_fallback,
    )
    try:
        sample = next(sample_iter)
    except StopIteration as exc:
        raise RuntimeError(f"No samples were loaded from: {train_dir}") from exc

    print("Sample preview:")
    print(
        json.dumps(
            build_metadata(
                sample=sample,
                output_root=output_root,
                sample_index=0,
                subdir_size=subdir_size,
                index_width=index_width,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


def resolve_output_paths(
    output_root: Path,
    sample_index: int,
    subdir_size: int,
    index_width: int,
) -> tuple[Path, Path, Path, Path]:
    subdir = output_root / f"sub{sample_index // subdir_size:04d}"
    stem = f"img_{sample_index:0{index_width}d}"
    png_path = subdir / f"{stem}.png"
    json_path = subdir / f"{stem}.json"
    rel_png_path = png_path.relative_to(output_root)
    return subdir, png_path, json_path, rel_png_path


def ensure_png_ready(image: Image.Image) -> Image.Image:
    if image.mode in {"RGB", "RGBA", "L", "LA"}:
        return image
    return image.convert("RGB")


def save_single_sample(
    sample: dict[str, Any],
    output_root: Path,
    sample_index: int,
    subdir_size: int,
    index_width: int,
) -> dict[str, Any]:
    subdir, png_path, json_path, _ = resolve_output_paths(
        output_root=output_root,
        sample_index=sample_index,
        subdir_size=subdir_size,
        index_width=index_width,
    )
    subdir.mkdir(parents=True, exist_ok=True)

    with Image.open(io.BytesIO(sample["img_content"])) as image:
        image = ensure_png_ready(image)
        image.save(png_path, format="PNG")

    metadata = build_metadata(
        sample=sample,
        output_root=output_root,
        sample_index=sample_index,
        subdir_size=subdir_size,
        index_width=index_width,
    )
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, separators=(",", ":"))

    return metadata


def drain_futures(
    futures: set[concurrent.futures.Future],
    completed: int,
    wait_for_all: bool,
) -> tuple[set[concurrent.futures.Future], int]:
    if not futures:
        return futures, completed

    return_when = (
        concurrent.futures.ALL_COMPLETED
        if wait_for_all
        else concurrent.futures.FIRST_COMPLETED
    )
    done, pending = concurrent.futures.wait(futures, return_when=return_when)
    for future in done:
        future.result()
        completed += 1
    return pending, completed


def confirm_continue(assume_yes: bool) -> bool:
    if assume_yes:
        return True

    if not sys.stdin.isatty():
        print("Non-interactive session detected, stop after preview. Use --yes to continue.")
        return False

    answer = input("Continue offloading all samples? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def offload_dataset(args: argparse.Namespace) -> int:
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    completed = 0
    last_logged_completed = 0
    futures: set[concurrent.futures.Future] = set()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.write_workers
    ) as executor:
        for sample_index, sample in enumerate(
            iter_samples(
                train_dir=args.rs5m_root,
                num_workers=args.loader_workers,
                batch_size=args.batch_size,
                prefer_tar_fallback=args.prefer_tar_fallback,
            )
        ):
            if args.max_samples is not None and sample_index >= args.max_samples:
                break

            futures.add(
                executor.submit(
                    save_single_sample,
                    sample=sample,
                    output_root=output_root,
                    sample_index=sample_index,
                    subdir_size=args.subdir_size,
                    index_width=args.index_width,
                )
            )

            if len(futures) >= args.max_pending_writes:
                futures, completed = drain_futures(
                    futures=futures,
                    completed=completed,
                    wait_for_all=False,
                )

            if completed - last_logged_completed >= args.log_interval:
                elapsed = time.time() - start_time
                print(
                    f"Completed {completed} samples in {elapsed:.1f}s "
                    f"({completed / max(elapsed, 1e-6):.2f} samples/s)."
                )
                last_logged_completed = completed

        futures, completed = drain_futures(
            futures=futures,
            completed=completed,
            wait_for_all=True,
        )

    elapsed = time.time() - start_time
    print(
        f"Finished offloading {completed} samples into {output_root} "
        f"in {elapsed:.1f}s."
    )
    return completed


def validate_args(args: argparse.Namespace) -> None:
    if not args.rs5m_root.exists():
        raise FileNotFoundError(f"RS5M root does not exist: {args.rs5m_root}")
    if args.loader_workers < 0:
        raise ValueError("--loader-workers must be >= 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.write_workers <= 0:
        raise ValueError("--write-workers must be > 0")
    if args.subdir_size <= 0:
        raise ValueError("--subdir-size must be > 0")
    if args.index_width <= 0:
        raise ValueError("--index-width must be > 0")
    if args.max_pending_writes <= 0:
        raise ValueError("--max-pending-writes must be > 0")
    if args.log_interval <= 0:
        raise ValueError("--log-interval must be > 0")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be > 0 when provided")


def main() -> None:
    args = parse_args()
    if args.max_pending_writes is None:
        args.max_pending_writes = max(args.write_workers * 4, args.write_workers)
    validate_args(args)

    print_one_sample_preview(
        train_dir=args.rs5m_root,
        output_root=args.output_root,
        num_workers=args.loader_workers,
        batch_size=args.batch_size,
        subdir_size=args.subdir_size,
        index_width=args.index_width,
        prefer_tar_fallback=args.prefer_tar_fallback,
    )

    if args.preview_only:
        return

    if not confirm_continue(args.yes):
        return

    offload_dataset(args)


if __name__ == "__main__":
    main()
