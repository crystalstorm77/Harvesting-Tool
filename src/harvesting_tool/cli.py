# ============================================================
# SECTION A — Imports And Argument Parsing
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path

from harvesting_tool.detection import (
    DetectorSettings,
    Timecode,
    detect_candidate_clips,
    parse_chapter_range,
    write_cut_lists,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Harvest candidate clips from a screen-recorded art session.")
    parser.add_argument("video_path", type=Path, help="Path to the input screen-recorded video.")
    parser.add_argument("--chapter-start", required=True, help="Chapter start timecode in HH:MM:SS:FF at 30 FPS.")
    parser.add_argument("--chapter-end", required=True, help="Chapter end timecode in HH:MM:SS:FF at 30 FPS.")
    parser.add_argument("--min-harvest", required=True, help="Minimum harvested duration in HH:MM:SS:FF.")
    parser.add_argument("--max-harvest", required=True, help="Maximum harvested duration in HH:MM:SS:FF.")
    parser.add_argument("--max-clip-length", required=True, help="Maximum clip length in HH:MM:SS:FF.")
    parser.add_argument("--lead-in-seconds", type=int, default=0, help="Lead-in seconds before detected activity.")
    parser.add_argument("--lead-in-frames", type=int, default=0, help="Lead-in frames at 30 FPS.")
    parser.add_argument("--tail-after-seconds", type=int, default=0, help="Tail-after seconds after detected activity.")
    parser.add_argument("--tail-after-frames", type=int, default=0, help="Tail-after frames at 30 FPS.")
    parser.add_argument("--output-stem", type=Path, required=True, help="Output path stem used for .txt and .json cut lists.")
    parser.add_argument("--sample-stride", type=int, default=3, help="Analyze every Nth frame for first-pass activity detection.")
    parser.add_argument("--activity-threshold", type=float, default=12.0, help="Per-pixel delta threshold for activity.")
    parser.add_argument("--active-pixel-ratio", type=float, default=0.015, help="Fraction of changed pixels needed to mark activity.")
    parser.add_argument("--min-burst", default="00:00:00:15", help="Minimum burst length in HH:MM:SS:FF.")
    return parser


# ============================================================
# SECTION B — Settings Construction And Execution
# ============================================================

def build_settings(args: argparse.Namespace) -> DetectorSettings:
    return DetectorSettings(
        lead_in=Timecode.from_seconds_and_frames(args.lead_in_seconds, args.lead_in_frames),
        tail_after=Timecode.from_seconds_and_frames(args.tail_after_seconds, args.tail_after_frames),
        min_harvest=Timecode.from_hhmmssff(args.min_harvest),
        max_harvest=Timecode.from_hhmmssff(args.max_harvest),
        max_clip_length=Timecode.from_hhmmssff(args.max_clip_length),
        sample_stride=args.sample_stride,
        activity_threshold=args.activity_threshold,
        active_pixel_ratio=args.active_pixel_ratio,
        min_burst_length=Timecode.from_hhmmssff(args.min_burst),
    )


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    chapter_range = parse_chapter_range(args.chapter_start, args.chapter_end)
    settings = build_settings(args)
    clips = detect_candidate_clips(args.video_path, chapter_range, settings)
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    return write_cut_lists(args.output_stem, clips, settings)


# ============================================================
# SECTION C — Program Entry Point
# ============================================================

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        text_path, json_path = run(args)
    except Exception as exc:  # pragma: no cover - CLI surface
        parser.exit(status=1, message=f"Error: {exc}\n")

    parser.exit(status=0, message=f"Wrote cut lists to {text_path} and {json_path}\n")


if __name__ == "__main__":
    raise SystemExit(main())
