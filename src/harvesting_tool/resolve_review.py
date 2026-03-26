# ============================================================
# SECTION A - Imports And Constants
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.machinery
import importlib.util
import os
import sys
import types

FRAME_RATE = 30


# ============================================================
# SECTION B - Resolve Data Structures And Clip Planning
# ============================================================

RESOLVE_SCRIPT_MODULE_PATH = Path(
    os.getenv(
        "RESOLVE_SCRIPT_API",
        r"C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting",
    )
) / "Modules"


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


@dataclass(frozen=True)
class ResolveReviewOptions:
    timeline_name: str
    include_source_track: bool = True


@dataclass(frozen=True)
class ResolveReviewResult:
    timeline_name: str
    source_timeline_name: str
    clip_count: int
    source_track_included: bool


@dataclass(frozen=True)
class ResolveClipPlacement:
    start_frame: int
    end_frame: int
    record_frame: int
    track_index: int
    media_type: int = 1

    def to_clip_info(self, media_pool_item) -> dict[str, object]:
        return {
            "mediaPoolItem": media_pool_item,
            "startFrame": self.start_frame,
            "endFrame": self.end_frame,
            "recordFrame": self.record_frame,
            "trackIndex": self.track_index,
            "mediaType": self.media_type,
        }



def candidate_clip_from_dict(payload: dict[str, object]) -> CandidateClip:
    lead_in_payload = payload["lead_in"]
    tail_after_payload = payload["tail_after"]
    if not isinstance(lead_in_payload, dict) or not isinstance(tail_after_payload, dict):
        raise ValueError("Candidate clip payload is missing lead-in or tail-after timing data.")

    return CandidateClip(
        clip_index=int(payload["clip_index"]),
        source_path=str(payload["source_path"]),
        chapter_start=Timecode.from_hhmmssff(str(payload["chapter_start"])),
        chapter_end=Timecode.from_hhmmssff(str(payload["chapter_end"])),
        activity_start=Timecode.from_hhmmssff(str(payload["activity_start"])),
        activity_end=Timecode.from_hhmmssff(str(payload["activity_end"])),
        clip_start=Timecode.from_hhmmssff(str(payload["clip_start"])),
        clip_end=Timecode.from_hhmmssff(str(payload["clip_end"])),
        lead_in=Timecode.from_seconds_and_frames(
            int(lead_in_payload["seconds"]),
            int(lead_in_payload["frames"]),
        ),
        tail_after=Timecode.from_seconds_and_frames(
            int(tail_after_payload["seconds"]),
            int(tail_after_payload["frames"]),
        ),
    )



def load_review_payload(payload_path: Path) -> tuple[Path, ChapterRange, list[CandidateClip]]:
    import json

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    clip_payloads = payload.get("clips", [])
    if not isinstance(clip_payloads, list) or not clip_payloads:
        raise RuntimeError("The harvested JSON does not contain any candidate clips to assemble.")

    clips = [candidate_clip_from_dict(clip_payload) for clip_payload in clip_payloads]
    first_clip = clips[0]
    return Path(first_clip.source_path), ChapterRange(first_clip.chapter_start, first_clip.chapter_end), clips



def _to_inclusive_end(total_frames: int) -> int:
    return max(0, total_frames - 1)



def build_source_reference_placement(chapter_range: ChapterRange, track_index: int) -> ResolveClipPlacement:
    return ResolveClipPlacement(
        start_frame=chapter_range.start.total_frames,
        end_frame=_to_inclusive_end(chapter_range.end.total_frames),
        record_frame=0,
        track_index=track_index,
    )



def build_gap_preserved_placements(
    clips: list[CandidateClip],
    chapter_range: ChapterRange,
    track_index: int,
) -> list[ResolveClipPlacement]:
    placements: list[ResolveClipPlacement] = []
    chapter_origin = chapter_range.start.total_frames
    for clip in clips:
        placements.append(
            ResolveClipPlacement(
                start_frame=clip.clip_start.total_frames,
                end_frame=_to_inclusive_end(clip.clip_end.total_frames),
                record_frame=clip.clip_start.total_frames - chapter_origin,
                track_index=track_index,
            )
        )
    return placements



def build_packed_placements(clips: list[CandidateClip], track_index: int) -> list[ResolveClipPlacement]:
    placements: list[ResolveClipPlacement] = []
    current_record_frame = 0
    for clip in clips:
        placements.append(
            ResolveClipPlacement(
                start_frame=clip.clip_start.total_frames,
                end_frame=_to_inclusive_end(clip.clip_end.total_frames),
                record_frame=current_record_frame,
                track_index=track_index,
            )
        )
        current_record_frame += clip.duration.total_frames
    return placements


# ============================================================
# SECTION C - Resolve Connection And Timeline Discovery
# ============================================================

def load_resolve_module():
    try:
        import DaVinciResolveScript as bmd  # type: ignore
        return bmd
    except ImportError:
        pass

    if not RESOLVE_SCRIPT_MODULE_PATH.exists():
        raise RuntimeError(
            "Unable to locate the DaVinci Resolve scripting module path. "
            "Ensure Resolve is installed and its scripting API is available."
        )

    resolve_module_file = RESOLVE_SCRIPT_MODULE_PATH / "DaVinciResolveScript.py"
    if not resolve_module_file.exists():
        raise RuntimeError(
            "Unable to locate DaVinciResolveScript.py in the Resolve scripting modules directory."
        )

    if "imp" not in sys.modules:
        imp_shim = types.ModuleType("imp")

        def load_dynamic(name: str, path: str):
            loader = importlib.machinery.ExtensionFileLoader(name, path)
            spec = importlib.util.spec_from_file_location(name, path, loader=loader)
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to create a module spec for {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        imp_shim.load_dynamic = load_dynamic  # type: ignore[attr-defined]
        sys.modules["imp"] = imp_shim

    spec = importlib.util.spec_from_file_location("DaVinciResolveScript", resolve_module_file)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to create a Python import spec for DaVinci Resolve scripting.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["DaVinciResolveScript"] = module
    spec.loader.exec_module(module)
    loaded_module = sys.modules.get("DaVinciResolveScript", module)
    if not hasattr(loaded_module, "scriptapp"):
        raise RuntimeError("Loaded DaVinci Resolve scripting module does not expose scriptapp().")
    return loaded_module



def get_resolve_app():
    bmd = load_resolve_module()
    resolve = bmd.scriptapp("Resolve")
    if resolve is None:
        raise RuntimeError("Unable to connect to DaVinci Resolve. Make sure Resolve is running and scripting is enabled.")
    return resolve



def get_current_project_and_timeline(resolve):
    project_manager = resolve.GetProjectManager()
    if project_manager is None:
        raise RuntimeError("Unable to access the DaVinci Resolve project manager.")

    project = project_manager.GetCurrentProject()
    if project is None:
        raise RuntimeError("No DaVinci Resolve project is currently open.")

    timeline = project.GetCurrentTimeline()
    if timeline is None:
        raise RuntimeError("No active DaVinci Resolve timeline is currently selected.")

    return project, timeline



def find_source_media_pool_item(timeline, video_path: Path):
    expected_path = str(video_path.resolve()).lower()
    track_items = timeline.GetItemListInTrack("video", 1) or []
    if not track_items:
        raise RuntimeError("The active Resolve timeline does not contain any clips on video track 1.")

    matching_items = []
    for item in track_items:
        media_pool_item = item.GetMediaPoolItem()
        if media_pool_item is None:
            continue
        clip_path = str(media_pool_item.GetClipProperty("File Path") or "").lower()
        if clip_path == expected_path:
            matching_items.append((item, media_pool_item))

    if len(matching_items) == 1:
        return matching_items[0][1]
    if len(matching_items) > 1:
        longest = max(matching_items, key=lambda pair: pair[0].GetEnd() - pair[0].GetStart())
        return longest[1]

    if len(track_items) == 1:
        media_pool_item = track_items[0].GetMediaPoolItem()
        if media_pool_item is None:
            raise RuntimeError("The only clip on video track 1 is not backed by a media pool item.")
        return media_pool_item

    raise RuntimeError(
        "Could not identify the source media on video track 1. "
        "Place the raw footage on track 1 or ensure it matches the input video path."
    )


# ============================================================
# SECTION D - Resolve Review Timeline Assembly
# ============================================================

def ensure_video_track_count(timeline, required_count: int) -> None:
    current_count = timeline.GetTrackCount("video")
    while current_count < required_count:
        if not timeline.AddTrack("video"):
            raise RuntimeError("Failed to add a required video track to the Resolve review timeline.")
        current_count += 1



def append_clip_infos(media_pool, clip_infos: list[dict[str, object]]) -> None:
    if not clip_infos:
        return
    appended = media_pool.AppendToTimeline(clip_infos)
    if not appended:
        raise RuntimeError("Failed to append harvested clips to the Resolve review timeline.")



def create_review_timeline(
    video_path: Path,
    chapter_range: ChapterRange,
    clips: list[CandidateClip],
    options: ResolveReviewOptions,
    resolve_app=None,
) -> ResolveReviewResult:
    resolve = resolve_app if resolve_app is not None else get_resolve_app()
    project, source_timeline = get_current_project_and_timeline(resolve)
    media_pool = project.GetMediaPool()
    if media_pool is None:
        raise RuntimeError("Unable to access the Resolve media pool.")

    source_media_pool_item = find_source_media_pool_item(source_timeline, video_path)
    review_timeline = media_pool.CreateEmptyTimeline(options.timeline_name)
    if review_timeline is None:
        raise RuntimeError(f"Failed to create Resolve review timeline '{options.timeline_name}'.")
    if not project.SetCurrentTimeline(review_timeline):
        raise RuntimeError("Failed to switch Resolve to the newly created review timeline.")

    required_tracks = 3 if options.include_source_track else 2
    ensure_video_track_count(review_timeline, required_tracks)

    track_offset = 1
    if options.include_source_track:
        review_timeline.SetTrackName("video", 1, "Source Chapter")
        source_reference = build_source_reference_placement(chapter_range, track_index=1)
        append_clip_infos(media_pool, [source_reference.to_clip_info(source_media_pool_item)])
        track_offset = 2

    review_timeline.SetTrackName("video", track_offset, "Harvested With Gaps")
    review_timeline.SetTrackName("video", track_offset + 1, "Harvested Packed")

    gap_preserved_infos = [
        placement.to_clip_info(source_media_pool_item)
        for placement in build_gap_preserved_placements(clips, chapter_range, track_offset)
    ]
    packed_infos = [
        placement.to_clip_info(source_media_pool_item)
        for placement in build_packed_placements(clips, track_offset + 1)
    ]
    append_clip_infos(media_pool, gap_preserved_infos)
    append_clip_infos(media_pool, packed_infos)

    return ResolveReviewResult(
        timeline_name=options.timeline_name,
        source_timeline_name=source_timeline.GetName(),
        clip_count=len(clips),
        source_track_included=options.include_source_track,
    )
