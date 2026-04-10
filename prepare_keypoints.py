from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from signer_model import SignTranslatorModel

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv"}


def find_video_file(videos_dir: Path, attachment_id: str) -> Path | None:
    for path in videos_dir.rglob("*"):
        if path.is_file() and path.stem == attachment_id and path.suffix.lower() in VIDEO_EXTENSIONS:
            return path
    return None


def read_annotations(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "cp1251"):
        try:
            return pd.read_csv(path, sep="\t", encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, sep="\t")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MediaPipe hand keypoints from raw videos.")
    parser.add_argument("--annotations", type=Path, default=Path("annotations.csv"))
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("keypoints_all"))
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of samples to process.")
    parser.add_argument("--use-begin-end", action="store_true", help="Use begin/end frame range from annotations.csv.")
    args = parser.parse_args()

    annotations = read_annotations(args.annotations)
    annotations["attachment_id"] = annotations["attachment_id"].astype(str)

    translator = SignTranslatorModel(annotations_path=args.annotations)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    saved = 0
    skipped_missing = 0
    skipped_existing = 0

    for _, row in annotations.iterrows():
        attachment_id = str(row["attachment_id"])
        output_path = args.output_dir / f"{attachment_id}.npy"

        if output_path.exists():
            skipped_existing += 1
            continue

        video_path = find_video_file(args.videos_dir, attachment_id)
        if video_path is None:
            skipped_missing += 1
            continue

        start_frame = int(row["begin"]) if args.use_begin_end and pd.notna(row.get("begin")) else None
        end_frame = int(row["end"]) if args.use_begin_end and pd.notna(row.get("end")) else None

        try:
            keypoints = translator.extract_keypoints_from_video(video_path, start_frame, end_frame)
            np.save(output_path, keypoints)
            saved += 1
        except Exception as exc:
            print(f"Failed for {attachment_id}: {exc}")

        processed += 1
        if args.limit and processed >= args.limit:
            break

    print(f"Saved: {saved}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Skipped missing videos: {skipped_missing}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
