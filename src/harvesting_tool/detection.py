# ============================================================
# SECTION A - Imports And Constants
# ============================================================

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import csv
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

FRAME_RATE = 30
CANVAS_LEFT_RATIO = 0.18
CANVAS_TOP_RATIO = 0.08
CANVAS_RIGHT_RATIO = 0.82
CANVAS_BOTTOM_RATIO = 0.92
GRID_ROWS = 12
GRID_COLUMNS = 12
BLOCK_ACTIVITY_RATIO = 0.01
MAX_ACTIVE_BLOCKS = 28
MAX_ACTIVE_RATIO = 0.18
PROGRESS_STEP_PERCENT = 5
TOTAL_GRID_BLOCKS = GRID_ROWS * GRID_COLUMNS
BACKTRACK_BUFFER_SAMPLES = 20
END_INACTIVE_SAMPLES = 3
PERSISTENT_MOTION_RATIO = 0.55
SECONDARY_PERSISTENT_MOTION_RATIO = 0.35

ART_STATE_LEFT_RATIO = 0.20
ART_STATE_TOP_RATIO = 0.12
ART_STATE_RIGHT_RATIO = 0.80
ART_STATE_BOTTOM_RATIO = 0.88
ART_STATE_VALIDATION_WINDOW_FRAMES = 15 * FRAME_RATE
ART_STATE_BASELINE_MAX_SAMPLES = 15
ART_STATE_MIN_RATIO = 0.015
ART_STATE_MIN_BLOCKS = 18
ART_STATE_REVEAL_WINDOW_FRAMES = 3 * FRAME_RATE
OVERLAY_COMPACT_BLOCKS = 18
OVERLAY_POST_INSTABILITY_RATIO = 0.008
VALIDATION_MERGE_GAP_FRAMES = FRAME_RATE
# ============================================================
# SECTION B - Timecode And Clip Data Structures
# ============================================================

@dataclass(frozen=True)
class Timecode:
    total_frames: int

    @classmethod
    def from_hhmmssff(cls, value: str) -> "Timecode":
        parts = value.split(":")
        if len(parts) != 4:
            raise ValueError(f"Invalid timecode '{value}'. Expected HH:MM:SS:FF.")

        hours, minutes, seconds, frames = (int(part) for part in parts)
        if minutes < 0 or minutes >= 60 or seconds < 0 or seconds >= 60:
            raise ValueError(f"Invalid timecode '{value}'. Minutes and seconds must be 0-59.")
        if frames < 0 or frames >= FRAME_RATE:
            raise ValueError(
                f"Invalid timecode '{value}'. Frames must be 0-{FRAME_RATE - 1} at {FRAME_RATE} FPS."
            )

        total_frames = (((hours * 60) + minutes) * 60 + seconds) * FRAME_RATE + frames
        return cls(total_frames=total_frames)

    @classmethod
    def from_seconds_and_frames(cls, seconds: int, frames: int) -> "Timecode":
        if seconds < 0:
            raise ValueError("Seconds must be non-negative.")
        if frames < 0 or frames >= FRAME_RATE:
            raise ValueError(f"Frames must be 0-{FRAME_RATE - 1} at {FRAME_RATE} FPS.")
        return cls(total_frames=(seconds * FRAME_RATE) + frames)

    def to_hhmmssff(self) -> str:
        total_seconds, frames = divmod(self.total_frames, FRAME_RATE)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"

    def to_seconds_frames(self) -> dict[str, int]:
        seconds, frames = divmod(self.total_frames, FRAME_RATE)
        return {"seconds": seconds, "frames": frames}


@dataclass(frozen=True)
class ChapterRange:
    start: Timecode
    end: Timecode

    def __post_init__(self) -> None:
        if self.end.total_frames <= self.start.total_frames:
            raise ValueError("Chapter end must be after chapter start.")


@dataclass(frozen=True)
class CandidateClip:
    clip_index: int
    source_path: str
    chapter_start: Timecode
    chapter_end: Timecode
    activity_start: Timecode
    activity_end: Timecode
    clip_start: Timecode
    clip_end: Timecode
    lead_in: Timecode
    tail_after: Timecode

    @property
    def duration(self) -> Timecode:
        return Timecode(total_frames=self.clip_end.total_frames - self.clip_start.total_frames)

    def to_dict(self) -> dict[str, object]:
        return {
            "clip_index": self.clip_index,
            "source_path": self.source_path,
            "chapter_start": self.chapter_start.to_hhmmssff(),
            "chapter_end": self.chapter_end.to_hhmmssff(),
            "activity_start": self.activity_start.to_hhmmssff(),
            "activity_end": self.activity_end.to_hhmmssff(),
            "clip_start": self.clip_start.to_hhmmssff(),
            "clip_end": self.clip_end.to_hhmmssff(),
            "duration": self.duration.to_hhmmssff(),
            "lead_in": self.lead_in.to_seconds_frames(),
            "tail_after": self.tail_after.to_seconds_frames(),
        }


@dataclass(frozen=True)
class DetectorSettings:
    lead_in: Timecode
    tail_after: Timecode
    min_harvest: Timecode
    max_harvest: Timecode
    min_clip_length: Timecode
    max_clip_length: Timecode
    pause_threshold: Timecode
    sample_stride: int = 3
    activity_threshold: float = 12.0
    active_pixel_ratio: float = 0.015
    min_burst_length: Timecode = Timecode(total_frames=15)

    def __post_init__(self) -> None:
        if self.max_harvest.total_frames < self.min_harvest.total_frames:
            raise ValueError("Maximum harvest duration must be at least the minimum harvest duration.")
        if self.min_clip_length.total_frames <= 0:
            raise ValueError("Minimum clip length must be greater than zero.")
        if self.max_clip_length.total_frames < self.min_clip_length.total_frames:
            raise ValueError("Maximum clip length must be at least the minimum clip length.")
        if self.pause_threshold.total_frames < 0:
            raise ValueError("Pause threshold must be non-negative.")
        if self.sample_stride <= 0:
            raise ValueError("Sample stride must be greater than zero.")


@dataclass(frozen=True)
class SampleDebugRow:
    frame_index: int
    timecode: str
    adjacent_change_score: float
    persistent_change_score: float
    locality_score: float
    global_change_score: float
    enter_active: bool
    remain_active: bool
    active_state: bool
    micro_event_marker: str
    notes: str


@dataclass
class DetectionDebugBundle:
    sampled_frames: list[SampleDebugRow] = field(default_factory=list)
    micro_events: list[dict[str, object]] = field(default_factory=list)
    merged_bursts: list[dict[str, object]] = field(default_factory=list)
    final_candidate_clips: list[dict[str, object]] = field(default_factory=list)


# ============================================================
# SECTION C - Chapter Parsing And Clip Serialization
# ============================================================

def parse_chapter_range(start: str, end: str) -> ChapterRange:
    return ChapterRange(start=Timecode.from_hhmmssff(start), end=Timecode.from_hhmmssff(end))


def format_cut_list_text(clips: Iterable[CandidateClip]) -> str:
    clip_list = list(clips)
    if not clip_list:
        return "No candidate clips detected."

    lines = []
    for clip in clip_list:
        lines.extend(
            [
                f"Clip {clip.clip_index}",
                f"  Source: {clip.source_path}",
                f"  Chapter: {clip.chapter_start.to_hhmmssff()} -> {clip.chapter_end.to_hhmmssff()}",
                f"  Activity: {clip.activity_start.to_hhmmssff()} -> {clip.activity_end.to_hhmmssff()}",
                f"  Clip: {clip.clip_start.to_hhmmssff()} -> {clip.clip_end.to_hhmmssff()}",
                f"  Duration: {clip.duration.to_hhmmssff()}",
                f"  Lead-in: {clip.lead_in.to_seconds_frames()['seconds']}s {clip.lead_in.to_seconds_frames()['frames']}f",
                f"  Tail-after: {clip.tail_after.to_seconds_frames()['seconds']}s {clip.tail_after.to_seconds_frames()['frames']}f",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def build_cut_list_payload(clips: Iterable[CandidateClip], settings: DetectorSettings) -> dict[str, object]:
    clip_list = list(clips)
    total_frames = sum(clip.duration.total_frames for clip in clip_list)
    return {
        "frame_rate": FRAME_RATE,
        "requested": {
            "min_harvest": settings.min_harvest.to_hhmmssff(),
            "max_harvest": settings.max_harvest.to_hhmmssff(),
            "min_clip_length": settings.min_clip_length.to_hhmmssff(),
            "max_clip_length": settings.max_clip_length.to_hhmmssff(),
            "pause_threshold": settings.pause_threshold.to_hhmmssff(),
            "lead_in": settings.lead_in.to_seconds_frames(),
            "tail_after": settings.tail_after.to_seconds_frames(),
        },
        "actual": {
            "clip_count": len(clip_list),
            "harvested_duration": Timecode(total_frames=total_frames).to_hhmmssff(),
            "met_minimum": total_frames >= settings.min_harvest.total_frames,
        },
        "clips": [clip.to_dict() for clip in clip_list],
    }


def write_cut_lists(output_stem: Path, clips: Iterable[CandidateClip], settings: DetectorSettings) -> tuple[Path, Path]:
    clip_list = list(clips)
    text_path = output_stem.with_suffix(".txt")
    json_path = output_stem.with_suffix(".json")
    text_path.write_text(format_cut_list_text(clip_list) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(build_cut_list_payload(clip_list, settings), indent=2), encoding="utf-8")
    return text_path, json_path


def write_debug_artifacts(debug_stem: Path, debug_bundle: DetectionDebugBundle) -> dict[str, Path]:
    debug_stem.parent.mkdir(parents=True, exist_ok=True)
    frames_path = debug_stem.with_name(f"{debug_stem.name}_frames.csv")
    micro_events_path = debug_stem.with_name(f"{debug_stem.name}_micro_events.json")
    bursts_path = debug_stem.with_name(f"{debug_stem.name}_bursts.json")
    clips_path = debug_stem.with_name(f"{debug_stem.name}_candidate_clips.json")

    with frames_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_index",
                "timecode",
                "adjacent_change_score",
                "persistent_change_score",
                "locality_score",
                "global_change_score",
                "enter_active",
                "remain_active",
                "active_state",
                "micro_event_marker",
                "notes",
            ],
        )
        writer.writeheader()
        for row in debug_bundle.sampled_frames:
            writer.writerow(
                {
                    "frame_index": row.frame_index,
                    "timecode": row.timecode,
                    "adjacent_change_score": f"{row.adjacent_change_score:.6f}",
                    "persistent_change_score": f"{row.persistent_change_score:.6f}",
                    "locality_score": f"{row.locality_score:.6f}",
                    "global_change_score": f"{row.global_change_score:.6f}",
                    "enter_active": str(row.enter_active),
                    "remain_active": str(row.remain_active),
                    "active_state": str(row.active_state),
                    "micro_event_marker": row.micro_event_marker,
                    "notes": row.notes,
                }
            )

    micro_events_path.write_text(json.dumps(debug_bundle.micro_events, indent=2), encoding="utf-8")
    bursts_path.write_text(json.dumps(debug_bundle.merged_bursts, indent=2), encoding="utf-8")
    clips_path.write_text(json.dumps(debug_bundle.final_candidate_clips, indent=2), encoding="utf-8")
    return {
        "frames_csv": frames_path,
        "micro_events_json": micro_events_path,
        "bursts_json": bursts_path,
        "candidate_clips_json": clips_path,
    }


# ============================================================
# SECTION D - Burst Normalization And Clip Construction
# ============================================================

def normalize_activity_bursts(
    bursts: Iterable[tuple[int, int]],
    minimum_length: Timecode,
) -> list[tuple[int, int]]:
    normalized: list[tuple[int, int]] = []
    for start_frame, end_frame in bursts:
        if end_frame <= start_frame:
            continue
        if (end_frame - start_frame) < minimum_length.total_frames:
            continue
        normalized.append((start_frame, end_frame))
    return normalized


def merge_activity_bursts(
    bursts: Iterable[tuple[int, int]],
    pause_threshold: Timecode,
) -> list[tuple[int, int]]:
    burst_list = sorted(bursts)
    if not burst_list:
        return []

    merged: list[tuple[int, int]] = [burst_list[0]]
    for start_frame, end_frame in burst_list[1:]:
        previous_start, previous_end = merged[-1]
        if start_frame - previous_end <= pause_threshold.total_frames:
            merged[-1] = (previous_start, max(previous_end, end_frame))
        else:
            merged.append((start_frame, end_frame))
    return merged


def expand_clip_window(
    chunk_start: int,
    chunk_end: int,
    chapter_range: ChapterRange,
    settings: DetectorSettings,
) -> tuple[int, int]:
    clip_start = max(chapter_range.start.total_frames, chunk_start - settings.lead_in.total_frames)
    clip_end = min(chapter_range.end.total_frames, chunk_end + settings.tail_after.total_frames)

    current_duration = clip_end - clip_start
    if current_duration >= settings.min_clip_length.total_frames:
        return clip_start, clip_end

    extra_needed = settings.min_clip_length.total_frames - current_duration
    extra_before = extra_needed // 2
    extra_after = extra_needed - extra_before

    clip_start = max(chapter_range.start.total_frames, clip_start - extra_before)
    clip_end = min(chapter_range.end.total_frames, clip_end + extra_after)

    current_duration = clip_end - clip_start
    if current_duration < settings.min_clip_length.total_frames:
        remaining = settings.min_clip_length.total_frames - current_duration
        extend_before = min(remaining, clip_start - chapter_range.start.total_frames)
        clip_start -= extend_before
        remaining -= extend_before
        clip_end = min(chapter_range.end.total_frames, clip_end + remaining)

    return clip_start, clip_end


def build_candidate_clips(
    source_path: str,
    chapter_range: ChapterRange,
    bursts: Iterable[tuple[int, int]],
    settings: DetectorSettings,
    debug_bundle: DetectionDebugBundle | None = None,
) -> list[CandidateClip]:
    clips: list[CandidateClip] = []
    normalized_bursts = normalize_activity_bursts(bursts, settings.min_burst_length)
    merged_bursts = merge_activity_bursts(normalized_bursts, settings.pause_threshold)
    if debug_bundle is not None and not debug_bundle.merged_bursts:
        debug_bundle.merged_bursts = [
            {
                "burst_index": index + 1,
                "start": Timecode(total_frames=start_frame).to_hhmmssff(),
                "end": Timecode(total_frames=end_frame).to_hhmmssff(),
                "duration": Timecode(total_frames=end_frame - start_frame).to_hhmmssff(),
            }
            for index, (start_frame, end_frame) in enumerate(merged_bursts)
        ]

    content_limit = settings.max_clip_length.total_frames - settings.lead_in.total_frames - settings.tail_after.total_frames
    if content_limit <= 0:
        raise ValueError("Maximum clip length must be greater than lead-in plus tail-after.")

    next_index = 1
    for activity_start_frame, activity_end_frame in merged_bursts:
        chunk_start = activity_start_frame
        while chunk_start < activity_end_frame:
            chunk_end = min(chunk_start + content_limit, activity_end_frame)
            clip_start_frame, clip_end_frame = expand_clip_window(chunk_start, chunk_end, chapter_range, settings)

            clip = CandidateClip(
                clip_index=next_index,
                source_path=source_path,
                chapter_start=chapter_range.start,
                chapter_end=chapter_range.end,
                activity_start=Timecode(total_frames=chunk_start),
                activity_end=Timecode(total_frames=chunk_end),
                clip_start=Timecode(total_frames=clip_start_frame),
                clip_end=Timecode(total_frames=clip_end_frame),
                lead_in=Timecode(total_frames=chunk_start - clip_start_frame),
                tail_after=Timecode(total_frames=clip_end_frame - chunk_end),
            )
            clips.append(clip)
            next_index += 1
            chunk_start = chunk_end

    trimmed: list[CandidateClip] = []
    accumulated_frames = 0
    for clip in clips:
        clip_duration = clip.duration.total_frames
        if accumulated_frames + clip_duration > settings.max_harvest.total_frames:
            break
        trimmed.append(clip)
        accumulated_frames += clip_duration

    if debug_bundle is not None:
        debug_bundle.final_candidate_clips = [clip.to_dict() for clip in trimmed]
    return trimmed
# ============================================================
# SECTION E - First-Pass Video Activity Detection
# ============================================================

def compute_enter_ratio_threshold(settings: DetectorSettings) -> float:
    return min(settings.active_pixel_ratio * 0.02, 0.00035)


def compute_remain_ratio_threshold(settings: DetectorSettings) -> float:
    return compute_enter_ratio_threshold(settings) * 0.4


def compute_weak_ratio_threshold(settings: DetectorSettings) -> float:
    return compute_enter_ratio_threshold(settings) * 0.18


def is_weak_art_change_signal(
    adjacent_ratio: float,
    persistent_ratio: float,
    adjacent_blocks: int,
    persistent_blocks: int,
    settings: DetectorSettings,
) -> bool:
    return (
        persistent_ratio >= compute_weak_ratio_threshold(settings)
        and persistent_blocks >= 1
    ) or (
        adjacent_ratio >= compute_weak_ratio_threshold(settings) * 4
        and adjacent_blocks >= 2
    )


def should_enter_active_state(
    adjacent_ratio: float,
    persistent_ratio: float,
    adjacent_blocks: int,
    persistent_blocks: int,
    settings: DetectorSettings,
) -> bool:
    enter_threshold = compute_enter_ratio_threshold(settings)
    return (
        persistent_ratio >= enter_threshold
        and persistent_ratio <= MAX_ACTIVE_RATIO
        and persistent_blocks >= 2
        and persistent_blocks <= MAX_ACTIVE_BLOCKS
        and adjacent_ratio <= MAX_ACTIVE_RATIO
    )


def should_remain_active_state(
    adjacent_ratio: float,
    persistent_ratio: float,
    adjacent_blocks: int,
    persistent_blocks: int,
    settings: DetectorSettings,
) -> bool:
    remain_threshold = compute_remain_ratio_threshold(settings)
    return (
        persistent_ratio >= remain_threshold
        and persistent_ratio <= MAX_ACTIVE_RATIO
        and persistent_blocks >= 1
        and persistent_blocks <= MAX_ACTIVE_BLOCKS
        and adjacent_ratio <= MAX_ACTIVE_RATIO
    )


def classify_activity_signal(
    adjacent_ratio: float,
    persistent_ratio: float,
    adjacent_blocks: int,
    persistent_blocks: int,
    settings: DetectorSettings,
) -> bool:
    return should_enter_active_state(
        adjacent_ratio=adjacent_ratio,
        persistent_ratio=persistent_ratio,
        adjacent_blocks=adjacent_blocks,
        persistent_blocks=persistent_blocks,
        settings=settings,
    )


def backtrack_event_start(recent_signals: list[tuple[int, bool]], fallback_frame: int) -> int:
    candidate_start = fallback_frame
    for frame_index, is_weak_signal in reversed(recent_signals):
        if not is_weak_signal:
            break
        candidate_start = frame_index
    return candidate_start


def emit_scan_progress(
    current_frame: int,
    chapter_range: ChapterRange,
    callback: Callable[[int], None] | None,
    last_percent: int,
) -> int:
    if callback is None:
        return last_percent

    chapter_frames = max(1, chapter_range.end.total_frames - chapter_range.start.total_frames)
    processed_frames = max(0, current_frame - chapter_range.start.total_frames)
    percent_complete = min(100, int((processed_frames / chapter_frames) * 100))
    if percent_complete >= last_percent + PROGRESS_STEP_PERCENT:
        rounded_percent = min(100, (percent_complete // PROGRESS_STEP_PERCENT) * PROGRESS_STEP_PERCENT)
        callback(rounded_percent)
        return rounded_percent
    return last_percent


def append_sample_debug_row(
    debug_bundle: DetectionDebugBundle | None,
    current_frame: int,
    adjacent_change_score: float,
    persistent_change_score: float,
    locality_score: float,
    global_change_score: float,
    enter_active: bool,
    remain_active: bool,
    active_state: bool,
    micro_event_marker: str,
    notes: str,
) -> None:
    if debug_bundle is None:
        return

    debug_bundle.sampled_frames.append(
        SampleDebugRow(
            frame_index=current_frame,
            timecode=Timecode(total_frames=current_frame).to_hhmmssff(),
            adjacent_change_score=adjacent_change_score,
            persistent_change_score=persistent_change_score,
            locality_score=locality_score,
            global_change_score=global_change_score,
            enter_active=enter_active,
            remain_active=remain_active,
            active_state=active_state,
            micro_event_marker=micro_event_marker,
            notes=notes,
        )
    )


def count_active_blocks(mask) -> int:
    grid_height = mask.shape[0] // GRID_ROWS
    grid_width = mask.shape[1] // GRID_COLUMNS
    if grid_height <= 0 or grid_width <= 0:
        return 0

    trimmed = mask[: grid_height * GRID_ROWS, : grid_width * GRID_COLUMNS]
    block_grid = trimmed.reshape(GRID_ROWS, grid_height, GRID_COLUMNS, grid_width)
    block_activity = block_grid.mean(axis=(1, 3)) / 255.0
    return int((block_activity >= BLOCK_ACTIVITY_RATIO).sum())


def extract_art_state_region(gray_frame):
    frame_height, frame_width = gray_frame.shape[:2]
    left = int(frame_width * ART_STATE_LEFT_RATIO)
    right = int(frame_width * ART_STATE_RIGHT_RATIO)
    top = int(frame_height * ART_STATE_TOP_RATIO)
    bottom = int(frame_height * ART_STATE_BOTTOM_RATIO)
    return gray_frame[top:bottom, left:right]


def build_persistent_change_mask(previous_gray, current_gray, next_gray, settings, cv2):
    prev_current_delta = cv2.absdiff(previous_gray, current_gray)
    prev_next_delta = cv2.absdiff(previous_gray, next_gray)
    current_next_delta = cv2.absdiff(current_gray, next_gray)

    _, prev_current_mask = cv2.threshold(prev_current_delta, settings.activity_threshold, 255, cv2.THRESH_BINARY)
    _, prev_next_mask = cv2.threshold(prev_next_delta, settings.activity_threshold, 255, cv2.THRESH_BINARY)
    _, current_next_mask = cv2.threshold(current_next_delta, settings.activity_threshold, 255, cv2.THRESH_BINARY)

    primary_persistent_mask = cv2.bitwise_and(prev_current_mask, prev_next_mask)
    secondary_persistent_mask = cv2.bitwise_and(prev_current_mask, current_next_mask)
    persistent_mask = cv2.addWeighted(
        primary_persistent_mask,
        PERSISTENT_MOTION_RATIO,
        secondary_persistent_mask,
        SECONDARY_PERSISTENT_MOTION_RATIO,
        0.0,
    )
    _, persistent_mask = cv2.threshold(persistent_mask, 127, 255, cv2.THRESH_BINARY)
    return prev_current_mask, persistent_mask


def build_art_state_change_mask(baseline_art_gray, current_art_gray, settings: DetectorSettings, cv2):
    art_delta = cv2.absdiff(baseline_art_gray, current_art_gray)
    _, art_mask = cv2.threshold(art_delta, settings.activity_threshold, 255, cv2.THRESH_BINARY)
    return art_mask


def compute_overlay_instability_ratio(
    changed_mask,
    post_baseline,
    post_samples: list[dict[str, object]],
    settings: DetectorSettings,
    cv2,
) -> float:
    if not post_samples:
        return 0.0

    instability_ratios: list[float] = []
    for sample in post_samples:
        sample_mask = build_art_state_change_mask(post_baseline, sample['art_gray'], settings, cv2)
        focused_mask = cv2.bitwise_and(sample_mask, changed_mask)
        instability_ratios.append(float(focused_mask.mean()) / 255.0)

    if not instability_ratios:
        return 0.0
    return sum(instability_ratios) / len(instability_ratios)


def select_representative_samples(
    samples: list[dict[str, object]],
    maximum_samples: int = ART_STATE_BASELINE_MAX_SAMPLES,
) -> list[dict[str, object]]:
    if len(samples) <= maximum_samples:
        return samples

    selected: list[dict[str, object]] = []
    last_index = len(samples) - 1
    for step_index in range(maximum_samples):
        sample_index = round((step_index / (maximum_samples - 1)) * last_index)
        selected.append(samples[sample_index])
    return selected


def build_median_baseline(samples: list[dict[str, object]]) -> object | None:
    if not samples:
        return None

    representative_samples = select_representative_samples(samples)
    stack = np.stack([sample['art_gray'] for sample in representative_samples], axis=0)
    baseline = np.median(stack, axis=0)
    return baseline.astype(np.uint8)


def collect_window_samples(
    sampled_frames: list[dict[str, object]],
    window_start: int,
    window_end: int,
) -> list[dict[str, object]]:
    return [
        sample
        for sample in sampled_frames
        if window_start <= int(sample['frame_index']) < window_end
    ]


def is_meaningful_update_signal(
    adjacent_ratio: float,
    persistent_ratio: float,
    settings: DetectorSettings,
) -> bool:
    return (
        persistent_ratio >= compute_weak_ratio_threshold(settings)
        or adjacent_ratio >= compute_weak_ratio_threshold(settings) * 4
    )


def find_effective_update_end(
    burst_start: int,
    burst_end: int,
    signal_rows: list[dict[str, float | int]],
    settings: DetectorSettings,
) -> int:
    burst_rows = [row for row in signal_rows if burst_start <= int(row["frame_index"]) <= burst_end]
    if not burst_rows:
        return burst_end

    effective_end = burst_end
    for row in reversed(burst_rows):
        adjacent_ratio = float(row["adjacent_ratio"])
        persistent_ratio = float(row["persistent_ratio"])
        if is_meaningful_update_signal(adjacent_ratio, persistent_ratio, settings):
            effective_end = int(row["frame_index"]) + settings.sample_stride
            break
    return min(burst_end, max(burst_start, effective_end))


def build_art_state_windows(
    burst_index: int,
    merged_bursts: list[tuple[int, int]],
    effective_update_end: int,
    chapter_range: ChapterRange,
    settings: DetectorSettings,
) -> tuple[tuple[int, int], tuple[int, int]]:
    burst_start, _ = merged_bursts[burst_index]
    previous_end = chapter_range.start.total_frames
    next_start = chapter_range.end.total_frames
    if burst_index > 0:
        previous_end = merged_bursts[burst_index - 1][1]
    if burst_index + 1 < len(merged_bursts):
        next_start = merged_bursts[burst_index + 1][0]

    pre_window_start = max(previous_end, burst_start - ART_STATE_VALIDATION_WINDOW_FRAMES)
    pre_window_end = max(pre_window_start, burst_start)
    post_window_start = min(next_start, effective_update_end + settings.sample_stride)
    post_window_end = min(next_start, effective_update_end + ART_STATE_VALIDATION_WINDOW_FRAMES)
    return (pre_window_start, pre_window_end), (post_window_start, post_window_end)


def build_reveal_window(
    burst_index: int,
    merged_bursts: list[tuple[int, int]],
    chapter_range: ChapterRange,
    effective_update_end: int,
    raw_burst_end: int,
) -> tuple[int, int]:
    reveal_start = max(raw_burst_end, effective_update_end)
    reveal_limit = chapter_range.end.total_frames
    if burst_index + 2 < len(merged_bursts):
        reveal_limit = merged_bursts[burst_index + 2][0]
    reveal_end = min(reveal_limit, reveal_start + ART_STATE_REVEAL_WINDOW_FRAMES)
    return reveal_start, max(reveal_start, reveal_end)


def summarize_burst_signal(
    burst_start: int,
    burst_end: int,
    signal_rows: list[dict[str, float | int]],
) -> tuple[float, float]:
    burst_rows = [row for row in signal_rows if burst_start <= int(row['frame_index']) <= burst_end]
    if not burst_rows:
        return 0.0, 0.0

    mean_adjacent_ratio = sum(float(row['adjacent_ratio']) for row in burst_rows) / len(burst_rows)
    peak_persistent_ratio = max(float(row['persistent_ratio']) for row in burst_rows)
    return mean_adjacent_ratio, peak_persistent_ratio


def validate_merged_burst_art_state(
    burst_index: int,
    merged_bursts: list[tuple[int, int]],
    sampled_frames: list[dict[str, object]],
    signal_rows: list[dict[str, float | int]],
    chapter_range: ChapterRange,
    settings: DetectorSettings,
    cv2,
) -> dict[str, object]:
    burst_start, burst_end = merged_bursts[burst_index]
    effective_update_end = find_effective_update_end(
        burst_start=burst_start,
        burst_end=burst_end,
        signal_rows=signal_rows,
        settings=settings,
    )
    (pre_window_start, pre_window_end), (post_window_start, post_window_end) = build_art_state_windows(
        burst_index=burst_index,
        merged_bursts=merged_bursts,
        effective_update_end=effective_update_end,
        chapter_range=chapter_range,
        settings=settings,
    )
    reveal_window_start, reveal_window_end = build_reveal_window(
        burst_index=burst_index,
        merged_bursts=merged_bursts,
        chapter_range=chapter_range,
        effective_update_end=effective_update_end,
        raw_burst_end=burst_end,
    )
    idle_hold_frames = max(0, burst_end - effective_update_end)

    pre_samples = collect_window_samples(sampled_frames, pre_window_start, pre_window_end)
    post_samples = collect_window_samples(sampled_frames, post_window_start, post_window_end)
    reveal_samples = collect_window_samples(sampled_frames, reveal_window_start, reveal_window_end)
    pre_baseline = build_median_baseline(pre_samples)
    post_baseline = build_median_baseline(post_samples)
    reveal_baseline = build_median_baseline(reveal_samples)
    mean_adjacent_ratio, peak_persistent_ratio = summarize_burst_signal(burst_start, burst_end, signal_rows)

    if pre_baseline is None or post_baseline is None:
        return {
            'validated': False,
            'art_state_ratio': 0.0,
            'art_state_blocks': 0,
            'overlay_instability_ratio': 0.0,
            'overlay_like': False,
            'reveal_ratio': 0.0,
            'reveal_recovery_ratio': 0.0,
            'effective_update_end': Timecode(total_frames=effective_update_end).to_hhmmssff(),
            'idle_hold_duration': Timecode(total_frames=idle_hold_frames).to_hhmmssff(),
            'trimmed_idle_hold': idle_hold_frames > 0,
            'mean_adjacent_ratio': mean_adjacent_ratio,
            'peak_persistent_ratio': peak_persistent_ratio,
            'pre_window_start': Timecode(total_frames=pre_window_start).to_hhmmssff(),
            'pre_window_end': Timecode(total_frames=pre_window_end).to_hhmmssff(),
            'post_window_start': Timecode(total_frames=post_window_start).to_hhmmssff(),
            'post_window_end': Timecode(total_frames=post_window_end).to_hhmmssff(),
            'reveal_window_start': Timecode(total_frames=reveal_window_start).to_hhmmssff(),
            'reveal_window_end': Timecode(total_frames=reveal_window_end).to_hhmmssff(),
            'pre_sample_count': len(pre_samples),
            'post_sample_count': len(post_samples),
            'reveal_sample_count': len(reveal_samples),
        }

    art_state_mask = build_art_state_change_mask(pre_baseline, post_baseline, settings, cv2)
    art_state_ratio = float(art_state_mask.mean()) / 255.0
    art_state_blocks = count_active_blocks(art_state_mask)
    overlay_instability_ratio = compute_overlay_instability_ratio(
        changed_mask=art_state_mask,
        post_baseline=post_baseline,
        post_samples=post_samples,
        settings=settings,
        cv2=cv2,
    )
    reveal_ratio = 0.0
    reveal_recovery_ratio = 1.0
    if reveal_baseline is not None:
        reveal_mask = build_art_state_change_mask(post_baseline, reveal_baseline, settings, cv2)
        focused_reveal_mask = cv2.bitwise_and(reveal_mask, art_state_mask)
        reveal_ratio = float(focused_reveal_mask.mean()) / 255.0
        reveal_recovery_mask = build_art_state_change_mask(pre_baseline, reveal_baseline, settings, cv2)
        focused_recovery_mask = cv2.bitwise_and(reveal_recovery_mask, art_state_mask)
        reveal_recovery_ratio = float(focused_recovery_mask.mean()) / 255.0
    overlay_like = (
        art_state_blocks <= OVERLAY_COMPACT_BLOCKS
        and (
            overlay_instability_ratio >= OVERLAY_POST_INSTABILITY_RATIO
            or (reveal_ratio >= art_state_ratio * 0.45 and reveal_recovery_ratio <= art_state_ratio * 0.35)
        )
    )
    required_ratio = max(
        ART_STATE_MIN_RATIO,
        compute_enter_ratio_threshold(settings) * 0.5,
        mean_adjacent_ratio * 0.08,
    )
    validated = (
        art_state_ratio >= required_ratio
        and art_state_blocks >= ART_STATE_MIN_BLOCKS
        and not overlay_like
    )
    return {
        'validated': validated,
        'art_state_ratio': art_state_ratio,
        'art_state_blocks': art_state_blocks,
        'overlay_instability_ratio': overlay_instability_ratio,
        'overlay_like': overlay_like,
        'reveal_ratio': reveal_ratio,
        'reveal_recovery_ratio': reveal_recovery_ratio,
        'effective_update_end': Timecode(total_frames=effective_update_end).to_hhmmssff(),
        'idle_hold_duration': Timecode(total_frames=idle_hold_frames).to_hhmmssff(),
        'trimmed_idle_hold': idle_hold_frames > 0,
        'mean_adjacent_ratio': mean_adjacent_ratio,
        'peak_persistent_ratio': peak_persistent_ratio,
        'pre_window_start': Timecode(total_frames=pre_window_start).to_hhmmssff(),
        'pre_window_end': Timecode(total_frames=pre_window_end).to_hhmmssff(),
        'post_window_start': Timecode(total_frames=post_window_start).to_hhmmssff(),
        'post_window_end': Timecode(total_frames=post_window_end).to_hhmmssff(),
        'reveal_window_start': Timecode(total_frames=reveal_window_start).to_hhmmssff(),
        'reveal_window_end': Timecode(total_frames=reveal_window_end).to_hhmmssff(),
        'pre_sample_count': len(pre_samples),
        'post_sample_count': len(post_samples),
        'reveal_sample_count': len(reveal_samples),
    }


def detect_activity_bursts(
    video_path: Path,
    chapter_range: ChapterRange,
    settings: DetectorSettings,
    progress_callback: Callable[[int], None] | None = None,
    debug_bundle: DetectionDebugBundle | None = None,
) -> list[tuple[int, int]]:
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(
            "OpenCV is required for first-pass video detection in this version. "
            "Install an OpenCV package in the runtime environment before running detection."
        ) from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    raw_bursts: list[tuple[int, int]] = []
    active_start: int | None = None
    active_end: int | None = None
    last_progress_percent = -PROGRESS_STEP_PERCENT
    inactive_samples = 0
    recent_signals: list[tuple[int, bool]] = []
    sampled_frames: deque[dict[str, object]] = deque(maxlen=3)
    sampled_frame_history: list[dict[str, object]] = []
    signal_rows: list[dict[str, float | int]] = []

    def process_sample_window(
        previous_sample: dict[str, object],
        current_sample: dict[str, object],
        next_sample: dict[str, object],
        active_start_value: int | None,
        active_end_value: int | None,
        inactive_samples_value: int,
    ) -> tuple[int | None, int | None, int]:
        previous_gray = previous_sample['gray']
        current_frame = int(current_sample['frame_index'])
        current_gray = current_sample['gray']
        next_frame = int(next_sample['frame_index'])
        next_gray = next_sample['gray']

        adjacent_mask, persistent_mask = build_persistent_change_mask(
            previous_gray=previous_gray,
            current_gray=current_gray,
            next_gray=next_gray,
            settings=settings,
            cv2=cv2,
        )
        adjacent_change_score = float(adjacent_mask.mean()) / 255.0
        persistent_change_score = float(persistent_mask.mean()) / 255.0
        adjacent_blocks = count_active_blocks(adjacent_mask)
        persistent_blocks = count_active_blocks(persistent_mask)
        locality_score = persistent_blocks / TOTAL_GRID_BLOCKS
        global_change_score = adjacent_change_score
        enter_active = False
        remain_active = False
        micro_event_marker = ''

        weak_signal = is_weak_art_change_signal(
            adjacent_ratio=adjacent_change_score,
            persistent_ratio=persistent_change_score,
            adjacent_blocks=adjacent_blocks,
            persistent_blocks=persistent_blocks,
            settings=settings,
        )
        recent_signals.append((current_frame, weak_signal))
        recent_signals[:] = recent_signals[-BACKTRACK_BUFFER_SAMPLES:]
        signal_rows.append(
            {
                'frame_index': current_frame,
                'adjacent_ratio': adjacent_change_score,
                'persistent_ratio': persistent_change_score,
            }
        )

        if active_start_value is None:
            if should_enter_active_state(
                adjacent_ratio=adjacent_change_score,
                persistent_ratio=persistent_change_score,
                adjacent_blocks=adjacent_blocks,
                persistent_blocks=persistent_blocks,
                settings=settings,
            ):
                enter_active = True
                micro_event_marker = 'start'
                active_start_value = backtrack_event_start(recent_signals, current_frame)
                active_end_value = next_frame + 1
                inactive_samples_value = 0
        else:
            if should_remain_active_state(
                adjacent_ratio=adjacent_change_score,
                persistent_ratio=persistent_change_score,
                adjacent_blocks=adjacent_blocks,
                persistent_blocks=persistent_blocks,
                settings=settings,
            ):
                remain_active = True
                micro_event_marker = 'continue'
                active_end_value = next_frame + 1
                inactive_samples_value = 0
            else:
                inactive_samples_value += 1
                if inactive_samples_value >= END_INACTIVE_SAMPLES:
                    raw_bursts.append((active_start_value, active_end_value if active_end_value is not None else current_frame))
                    active_start_value = None
                    active_end_value = None
                    inactive_samples_value = 0
                    micro_event_marker = 'end'

        notes = (
            f"adjacent_blocks={adjacent_blocks};persistent_blocks={persistent_blocks};"
            f"inactive_samples={inactive_samples_value}"
        )
        append_sample_debug_row(
            debug_bundle=debug_bundle,
            current_frame=current_frame,
            adjacent_change_score=adjacent_change_score,
            persistent_change_score=persistent_change_score,
            locality_score=locality_score,
            global_change_score=global_change_score,
            enter_active=enter_active,
            remain_active=remain_active,
            active_state=active_start_value is not None,
            micro_event_marker=micro_event_marker,
            notes=notes,
        )
        return active_start_value, active_end_value, inactive_samples_value

    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, chapter_range.start.total_frames)
        current_frame = chapter_range.start.total_frames

        while current_frame < chapter_range.end.total_frames:
            success, frame = capture.read()
            if not success:
                break

            last_progress_percent = emit_scan_progress(
                current_frame=current_frame,
                chapter_range=chapter_range,
                callback=progress_callback,
                last_percent=last_progress_percent,
            )

            if ((current_frame - chapter_range.start.total_frames) % settings.sample_stride) == 0:
                frame_height, frame_width = frame.shape[:2]
                left = int(frame_width * CANVAS_LEFT_RATIO)
                right = int(frame_width * CANVAS_RIGHT_RATIO)
                top = int(frame_height * CANVAS_TOP_RATIO)
                bottom = int(frame_height * CANVAS_BOTTOM_RATIO)
                canvas_frame = frame[top:bottom, left:right]

                gray = cv2.cvtColor(canvas_frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                sample = {
                    'frame_index': current_frame,
                    'gray': gray,
                    'art_gray': extract_art_state_region(gray),
                }
                sampled_frames.append(sample)
                sampled_frame_history.append(sample)

                if len(sampled_frames) == 3:
                    active_start, active_end, inactive_samples = process_sample_window(
                        previous_sample=sampled_frames[0],
                        current_sample=sampled_frames[1],
                        next_sample=sampled_frames[2],
                        active_start_value=active_start,
                        active_end_value=active_end,
                        inactive_samples_value=inactive_samples,
                    )

            current_frame += 1

        if len(sampled_frames) >= 2:
            previous_sample = sampled_frames[-2]
            current_sample = sampled_frames[-1]
            active_start, active_end, inactive_samples = process_sample_window(
                previous_sample=previous_sample,
                current_sample=current_sample,
                next_sample=current_sample,
                active_start_value=active_start,
                active_end_value=active_end,
                inactive_samples_value=inactive_samples,
            )

        if active_start is not None:
            raw_bursts.append((active_start, active_end if active_end is not None else chapter_range.end.total_frames))
    finally:
        capture.release()

    normalized_bursts = normalize_activity_bursts(raw_bursts, settings.min_burst_length)
    validation_bursts = merge_activity_bursts(normalized_bursts, Timecode(total_frames=VALIDATION_MERGE_GAP_FRAMES))
    validated_bursts: list[tuple[int, int]] = []

    if debug_bundle is not None:
        debug_bundle.micro_events = [
            {
                'micro_event_index': burst_index,
                'start': Timecode(total_frames=burst_start).to_hhmmssff(),
                'end': Timecode(total_frames=burst_end).to_hhmmssff(),
                'duration': Timecode(total_frames=burst_end - burst_start).to_hhmmssff(),
            }
            for burst_index, (burst_start, burst_end) in enumerate(normalized_bursts, start=1)
        ]
        debug_bundle.merged_bursts = []

    for burst_index, (burst_start, burst_end) in enumerate(validation_bursts, start=1):
        validation = validate_merged_burst_art_state(
            burst_index=burst_index - 1,
            merged_bursts=validation_bursts,
            sampled_frames=sampled_frame_history,
            signal_rows=signal_rows,
            chapter_range=chapter_range,
            settings=settings,
            cv2=cv2,
        )
        if debug_bundle is not None:
            debug_bundle.merged_bursts.append(
                {
                    'burst_index': burst_index,
                    'start': Timecode(total_frames=burst_start).to_hhmmssff(),
                    'end': Timecode(total_frames=burst_end).to_hhmmssff(),
                    'duration': Timecode(total_frames=burst_end - burst_start).to_hhmmssff(),
                    'validated': bool(validation['validated']),
                    'art_state_ratio': round(float(validation['art_state_ratio']), 6),
                    'art_state_blocks': int(validation['art_state_blocks']),
                    'overlay_instability_ratio': round(float(validation['overlay_instability_ratio']), 6),
                    'overlay_like': bool(validation['overlay_like']),
                    'reveal_ratio': round(float(validation['reveal_ratio']), 6),
                    'reveal_recovery_ratio': round(float(validation['reveal_recovery_ratio']), 6),
                      'effective_update_end': validation['effective_update_end'],
                      'idle_hold_duration': validation['idle_hold_duration'],
                      'trimmed_idle_hold': bool(validation['trimmed_idle_hold']),
                    'mean_adjacent_ratio': round(float(validation['mean_adjacent_ratio']), 6),
                    'peak_persistent_ratio': round(float(validation['peak_persistent_ratio']), 6),
                    'pre_window_start': validation['pre_window_start'],
                    'pre_window_end': validation['pre_window_end'],
                    'post_window_start': validation['post_window_start'],
                    'post_window_end': validation['post_window_end'],
                    'reveal_window_start': validation['reveal_window_start'],
                    'reveal_window_end': validation['reveal_window_end'],
                    'pre_sample_count': int(validation['pre_sample_count']),
                    'post_sample_count': int(validation['post_sample_count']),
                    'reveal_sample_count': int(validation['reveal_sample_count']),
                }
            )
        if bool(validation['validated']):
            validated_bursts.append((burst_start, burst_end))

    merged_validated_bursts = merge_activity_bursts(validated_bursts, settings.pause_threshold)

    if progress_callback is not None:
        progress_callback(100)
    return merged_validated_bursts


def detect_candidate_clips(
    video_path: Path,
    chapter_range: ChapterRange,
    settings: DetectorSettings,
    progress_callback: Callable[[int], None] | None = None,
    debug_bundle: DetectionDebugBundle | None = None,
) -> list[CandidateClip]:
    bursts = detect_activity_bursts(
        video_path,
        chapter_range,
        settings,
        progress_callback=progress_callback,
        debug_bundle=debug_bundle,
    )
    return build_candidate_clips(str(video_path), chapter_range, bursts, settings, debug_bundle=debug_bundle)













