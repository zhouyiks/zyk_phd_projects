#!/usr/bin/env python3
"""Extract a selected subdirectory from Satlas tar archives and delete each tar on success."""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath


ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz")


class ExtractionError(RuntimeError):
    """Raised when an archive cannot be extracted safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract only the selected image subdirectory from Satlas archives under a directory. "
            "Each archive is deleted only after a successful extraction."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="/media/disk5/dataset/satlas/dataset",
        help="Directory that contains the Satlas tar archives.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many archives.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Do not delete tar files after a successful extraction.",
    )
    parser.add_argument(
        "--keep-subdir",
        choices=("tci", "ir"),
        default="tci",
        help="Keep only this image subdirectory from each archive. Default: tci.",
    )
    return parser.parse_args()


def is_archive(path: Path) -> bool:
    return path.is_file() and path.name.endswith(ARCHIVE_SUFFIXES)


def list_archives(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if is_archive(path))


def has_target_segment(member_name: str, keep_subdir: str) -> bool:
    parts = PurePosixPath(member_name).parts
    return keep_subdir in parts


def safe_target_path(root: Path, member_name: str) -> Path:
    parts = PurePosixPath(member_name).parts
    if not parts:
        raise ExtractionError("Encountered an empty archive member name.")
    if parts[0] == "/":
        raise ExtractionError(f"Absolute archive path is not allowed: {member_name}")
    if ".." in parts:
        raise ExtractionError(f"Parent path traversal is not allowed: {member_name}")
    return root.joinpath(*parts)


def extract_selected_subdir(
    archive_path: Path, output_root: Path, keep_subdir: str
) -> tuple[int, int]:
    file_count = 0
    skipped_count = 0

    with tarfile.open(archive_path, mode="r|*") as archive:
        for member in archive:
            if not has_target_segment(member.name, keep_subdir):
                skipped_count += 1
                continue

            target_path = safe_target_path(output_root, member.name)

            if member.isdir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue

            if not member.isreg():
                skipped_count += 1
                continue

            extracted_file = archive.extractfile(member)
            if extracted_file is None:
                raise ExtractionError(f"Failed to read file member: {member.name}")

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with extracted_file, target_path.open("wb") as output_file:
                shutil.copyfileobj(extracted_file, output_file, length=1024 * 1024)
            file_count += 1

    if file_count == 0:
        raise ExtractionError(f"No {keep_subdir} files were found in the archive.")

    return file_count, skipped_count


def process_archive(
    archive_path: Path, output_root: Path, keep_archives: bool, keep_subdir: str
) -> bool:
    print(f"[START] {archive_path.name}", flush=True)

    try:
        file_count, skipped_count = extract_selected_subdir(
            archive_path, output_root, keep_subdir
        )
    except Exception as exc:
        print(f"[FAIL]  {archive_path.name}: {exc}", file=sys.stderr, flush=True)
        return False

    if keep_archives:
        print(
            f"[OK]    {archive_path.name}: extracted {file_count} {keep_subdir} files, "
            f"kept archive, skipped {skipped_count} non-{keep_subdir} members",
            flush=True,
        )
        return True

    archive_path.unlink()
    print(
        f"[OK]    {archive_path.name}: extracted {file_count} {keep_subdir} files, "
        f"deleted archive, skipped {skipped_count} non-{keep_subdir} members",
        flush=True,
    )
    return True


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()

    if not root.exists():
        print(f"Root directory does not exist: {root}", file=sys.stderr)
        return 1
    if not root.is_dir():
        print(f"Root path is not a directory: {root}", file=sys.stderr)
        return 1

    archives = list_archives(root)
    if args.limit is not None:
        archives = archives[: args.limit]

    if not archives:
        print(f"No archives found under {root}")
        return 0

    print(f"Found {len(archives)} archives under {root}", flush=True)

    success_count = 0
    failure_count = 0
    for archive_path in archives:
        if process_archive(archive_path, root, args.keep_archives, args.keep_subdir):
            success_count += 1
        else:
            failure_count += 1

    print(
        f"Finished: {success_count} succeeded, {failure_count} failed, "
        f"{len(archives)} total",
        flush=True,
    )
    return 0 if failure_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
