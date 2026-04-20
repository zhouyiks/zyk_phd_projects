#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <archive.tar[.gz|.bz2|.xz]> <output_dir>" >&2
  exit 1
fi

archive_path=$1
output_dir=$2

if [[ ! -f "$archive_path" ]]; then
  echo "Archive not found: $archive_path" >&2
  exit 1
fi

mkdir -p "$output_dir"

image_pattern='\.(jpg|jpeg|png|bmp|gif|webp|tif|tiff)$'
extracted_count=0
skipped_count=0

while IFS= read -r entry; do
  [[ -z "$entry" ]] && continue

  lower_entry=${entry,,}
  if [[ ! "$lower_entry" =~ $image_pattern ]]; then
    continue
  fi

  file_name=$(basename -- "$entry")
  target_path="$output_dir/$file_name"

  if [[ -e "$target_path" ]]; then
    echo "Skip existing: $file_name"
    ((skipped_count+=1))
    continue
  fi

  tar -xOf "$archive_path" "$entry" > "$target_path"
  echo "Extracted: $file_name"
  ((extracted_count+=1))
done < <(tar -tf "$archive_path")

echo "Done. Extracted: $extracted_count, skipped: $skipped_count"
