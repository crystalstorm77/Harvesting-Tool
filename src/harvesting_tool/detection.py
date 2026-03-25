# ============================================================
# SECTION A — Imports And Constants
# ============================================================

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable


FRAME_RATE = 30


# ============================================================
# SECTION B — Timecode And Clip Data Structures
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

    def clamp(self, minimum: "Timecode", maximum: "Timecode") -> "Timecode":
        return Timecode(total_frames=max(minimum.total_frames, min(self.total_frames, maximum.total_frames)))


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
    max_clip_length: Timecode
    sample_stride: int = 3
    activity_threshold: float = 12.0
    active_pixel_ratio: float = 0.015
    min_burst_length: Timecode = Timecode(total_frames=15)

    def __post_init__(self) -> None:
        if self.max_harvest.total_frames < self.min_harvest.total_frames:
            raise ValueError("Maximum harvest duration must be at least the minimum harvest duration.")
        if self.max_clip_length.total_frames <= 0:
            raise ValueError("Maximum clip length must be greater than zero.")
        if self.sample_stride <= 0:
            raise ValueError("Sample stride must be greater than zero.")


# ============================================================
# SECTION C — Chapter Parsing And Clip Serialization
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
            "max_clip_length": settings.max_clip_length.to_hhmmssff(),
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


# ============================================================
# SECTION D — Burst Normalization And Clip Construction
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


def build_candidate_clips(
    source_path: str,
    chapter_range: ChapterRange,
    bursts: Iterable[tuple[int, int]],
    settings: DetectorSettings,
) -> list[CandidateClip]:
    clips: list[CandidateClip] = []
    burst_windows = normalize_activity_bursts(bursts, settings.min_burst_length)
    content_limit = settings.max_clip_length.total_frames - settings.lead_in.total_frames - settings.tail_after.total_frames
    if content_limit <= 0:
        raise ValueError("Maximum clip length must be greater than lead-in plus tail-after.")

    next_index = 1
    for activity_start_frame, activity_end_frame in burst_windows:
        chunk_start = activity_start_frame
        while chunk_start < activity_end_frame:
            chunk_end = min(chunk_start + content_limit, activity_end_frame)
            clip_start_frame = max(chapter_range.start.total_frames, chunk_start - settings.lead_in.total_frames)
            clip_end_frame = min(chapter_range.end.total_frames, chunk_end + settings.tail_after.total_frames)

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
    return trimmed


# ============================================================
# SECTION E — First-Pass Video Activity Detection
# ============================================================

def detect_activity_bursts(video_path: Path, chapter_range: ChapterRange, settings: DetectorSettings) -> list[tuple[int, int]]:
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

    bursts: list[tuple[int, int]] = []
    active_start: int | None = None
    active_end: int | None = None
    previous_gray = None

    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, chapter_range.start.total_frames)
        current_frame = chapter_range.start.total_frames

        while current_frame < chapter_range.end.total_frames:
            success, frame = capture.read()
            if not success:
                break

            if ((current_frame - chapter_range.start.total_frames) % settings.sample_stride) != 0:
                current_frame += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if previous_gray is not None:
                frame_delta = cv2.absdiff(previous_gray, gray)
                _, thresholded = cv2.threshold(frame_delta, settings.activity_threshold, 255, cv2.THRESH_BINARY)
                active_ratio = float(thresholded.mean()) / 255.0

                if active_ratio >= settings.active_pixel_ratio:
                    if active_start is None:
                        active_start = max(chapter_range.start.total_frames, current_frame - settings.sample_stride)
                    active_end = current_frame + 1
                elif active_start is not None and active_end is not None:
                    bursts.append((active_start, active_end))
                    active_start = None
                    active_end = None

            previous_gray = gray
            current_frame += 1

        if active_start is not None and active_end is not None:
            bursts.append((active_start, active_end))
    finally:
        capture.release()

    return normalize_activity_bursts(bursts, settings.min_burst_length)


def detect_candidate_clips(
    video_path: Path,
    chapter_range: ChapterRange,
    settings: DetectorSettings,
) -> list[CandidateClip]:
    bursts = detect_activity_bursts(video_path, chapter_range, settings)
    return build_candidate_clips(str(video_path), chapter_range, bursts, settings)
