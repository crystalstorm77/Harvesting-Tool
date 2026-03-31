# ============================================================
# SECTION A - Imports And Argument Parsing
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from harvesting_tool.detection import (
    DetectionDebugBundle,
    DetectorSettings,
    Timecode,
    build_candidate_clips,
    detect_candidate_clips,
    parse_chapter_range,
    write_cut_lists,
    write_debug_artifacts,
)
from harvesting_tool.resolve_review import ResolveReviewOptions, ResolveReviewResult, create_review_timeline
from harvesting_tool.staged_detection import detect_staged_activity_ranges, write_staged_debug_artifacts



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Harvest candidate clips from a screen-recorded art session.")
    parser.add_argument("video_path", type=Path, help="Path to the input screen-recorded video.")
    parser.add_argument("--chapter-start", required=True, help="Chapter start timecode in HH:MM:SS:FF at 30 FPS.")
    parser.add_argument("--chapter-end", required=True, help="Chapter end timecode in HH:MM:SS:FF at 30 FPS.")
    parser.add_argument("--min-harvest", required=True, help="Minimum harvested duration in HH:MM:SS:FF.")
    parser.add_argument("--max-harvest", required=True, help="Maximum harvested duration in HH:MM:SS:FF.")
    parser.add_argument("--min-clip-length", default="00:00:00:15", help="Minimum clip length in HH:MM:SS:FF.")
    parser.add_argument("--max-clip-length", default="00:00:07:00", help="Maximum clip length in HH:MM:SS:FF.")
    parser.add_argument("--pause-threshold", default="00:00:05:00", help="Maximum gap between bursts before splitting clips.")
    parser.add_argument("--lead-in-seconds", type=int, default=0, help="Lead-in seconds before detected activity.")
    parser.add_argument("--lead-in-frames", type=int, default=2, help="Lead-in frames at 30 FPS.")
    parser.add_argument("--tail-after-seconds", type=int, default=0, help="Tail-after seconds after detected activity.")
    parser.add_argument("--tail-after-frames", type=int, default=4, help="Tail-after frames at 30 FPS.")
    parser.add_argument("--output-stem", type=Path, required=True, help="Output path stem used for .txt and .json cut lists.")
    parser.add_argument("--debug-stem", type=Path, help="Optional output path stem for detector diagnostic files.")
    parser.add_argument("--sample-stride", type=int, default=3, help="Analyze every Nth frame for first-pass activity detection.")
    parser.add_argument("--activity-threshold", type=float, default=12.0, help="Per-pixel delta threshold for activity.")
    parser.add_argument("--active-pixel-ratio", type=float, default=0.015, help="Fraction of changed pixels needed to mark activity.")
    parser.add_argument("--min-burst", default="00:00:00:10", help="Minimum burst length in HH:MM:SS:FF.")
    parser.add_argument(
        "--use-staged-detector",
        action="store_true",
        help="Run the new staged detector path instead of the current default detector. This is intended for side-by-side benchmarking while the staged detector is still being evaluated.",
    )
    parser.add_argument(
        "--resolve-review-timeline-name",
        help="Optional DaVinci Resolve review timeline name. When provided, accepted clips are also assembled into a new review timeline using the current active timeline as the source reference.",
    )
    parser.add_argument(
        "--resolve-skip-source-track",
        action="store_true",
        help="Do not add the chapter-length source reference clip to track 1 of the Resolve review timeline.",
    )
    return parser


# ============================================================
# SECTION B - Settings Construction And Execution
# ============================================================

def build_settings(args: argparse.Namespace) -> DetectorSettings:
    return DetectorSettings(
        lead_in=Timecode.from_seconds_and_frames(args.lead_in_seconds, args.lead_in_frames),
        tail_after=Timecode.from_seconds_and_frames(args.tail_after_seconds, args.tail_after_frames),
        min_harvest=Timecode.from_hhmmssff(args.min_harvest),
        max_harvest=Timecode.from_hhmmssff(args.max_harvest),
        min_clip_length=Timecode.from_hhmmssff(args.min_clip_length),
        max_clip_length=Timecode.from_hhmmssff(args.max_clip_length),
        pause_threshold=Timecode.from_hhmmssff(args.pause_threshold),
        sample_stride=args.sample_stride,
        activity_threshold=args.activity_threshold,
        active_pixel_ratio=args.active_pixel_ratio,
        min_burst_length=Timecode.from_hhmmssff(args.min_burst),
    )



def build_progress_reporter() -> Callable[[int], None]:
    last_reported_percent = -1

    def report(percent_complete: int) -> None:
        nonlocal last_reported_percent
        if percent_complete == last_reported_percent:
            return
        print(f"Scanning chapter: {percent_complete}%")
        last_reported_percent = percent_complete

    return report



def build_review_options(args: argparse.Namespace) -> ResolveReviewOptions | None:
    if not args.resolve_review_timeline_name:
        return None
    return ResolveReviewOptions(
        timeline_name=args.resolve_review_timeline_name,
        include_source_track=not args.resolve_skip_source_track,
    )



def run(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Path], ResolveReviewResult | None]:
    chapter_range = parse_chapter_range(args.chapter_start, args.chapter_end)
    settings = build_settings(args)
    progress_reporter = build_progress_reporter()

    if args.use_staged_detector:
        final_ranges, staged_debug_payload = detect_staged_activity_ranges(
            video_path=args.video_path,
            chapter_range=chapter_range,
            settings=settings,
            progress_callback=progress_reporter,
        )
        clips = build_candidate_clips(
            str(args.video_path),
            chapter_range,
            final_ranges,
            settings,
            trust_burst_boundaries=True,
        )
        staged_debug_payload['candidate_clips'] = [
            {
                'clip_index': index,
                'clip_start': clip.clip_start.to_hhmmssff(),
                'clip_end': clip.clip_end.to_hhmmssff(),
                'activity_start': clip.activity_start.to_hhmmssff(),
                'activity_end': clip.activity_end.to_hhmmssff(),
                'duration': clip.duration.to_hhmmssff(),
            }
            for index, clip in enumerate(clips, start=1)
        ]
        debug_paths: dict[str, Path] = {}
        if args.debug_stem:
            debug_paths = write_staged_debug_artifacts(args.debug_stem, staged_debug_payload)
    else:
        debug_bundle = DetectionDebugBundle() if args.debug_stem else None
        clips = detect_candidate_clips(
            args.video_path,
            chapter_range,
            settings,
            progress_callback=progress_reporter,
            debug_bundle=debug_bundle,
        )
        debug_paths = {}
        if args.debug_stem and debug_bundle is not None:
            debug_paths = write_debug_artifacts(args.debug_stem, debug_bundle)

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    text_path, json_path = write_cut_lists(args.output_stem, clips, settings)

    review_result: ResolveReviewResult | None = None
    review_options = build_review_options(args)
    if review_options is not None:
        review_result = create_review_timeline(
            video_path=args.video_path,
            chapter_range=chapter_range,
            clips=clips,
            options=review_options,
        )
    return text_path, json_path, debug_paths, review_result


# ============================================================
# SECTION C - Program Entry Point
# ============================================================

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        text_path, json_path, debug_paths, review_result = run(args)
    except Exception as exc:  # pragma: no cover - CLI surface
        parser.exit(status=1, message=f"Error: {exc}\n")

    lines = [f"Wrote cut lists to {text_path} and {json_path}"]
    if debug_paths:
        debug_summary = ", ".join(str(path) for path in debug_paths.values())
        lines.append(f"Wrote debug artifacts to {debug_summary}")
    if review_result is not None:
        lines.append(
            "Created Resolve review timeline "
            f"'{review_result.timeline_name}' from source timeline '{review_result.source_timeline_name}' "
            f"with {review_result.clip_count} accepted clips."
        )

    parser.exit(status=0, message="\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
