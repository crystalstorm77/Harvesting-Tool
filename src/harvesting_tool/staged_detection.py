from __future__ import annotations

# ============================================================
# SECTION A - Imports And Reused Detection Primitives
# ============================================================

from bisect import bisect_left
from collections import deque
from dataclasses import dataclass, replace
import json
import os
import numpy as np
from pathlib import Path
from typing import Callable, Iterable

from harvesting_tool.detection import (
    BACKTRACK_BUFFER_SAMPLES,
    CANVAS_BOTTOM_RATIO,
    CANVAS_LEFT_RATIO,
    CANVAS_RIGHT_RATIO,
    CANVAS_TOP_RATIO,
    END_INACTIVE_SAMPLES,
    FRAME_RATE,
    GRID_COLUMNS,
    GRID_ROWS,
    MAX_ACTIVE_BLOCKS,
    PROGRESS_STEP_PERCENT,
    TOTAL_GRID_BLOCKS,
    TRAIL_MASK_WINDOW,
    ChapterRange,
    DetectorSettings,
    Timecode,
    backtrack_event_start,
    build_art_state_change_mask,
    build_median_baseline,
    build_persistent_change_mask,
    build_trail_mask,
    collect_window_samples,
    count_active_blocks,
    emit_scan_progress,
    is_weak_art_change_signal,
    should_enter_active_state,
    should_remain_active_state,
)


# ============================================================
# SECTION B - Staged Detection Data Structures
# ============================================================

GridCoordinate = tuple[int, int]
STAGE1_OPEN_WINDOW_RECORDS = 3
STAGE1_OPEN_MIN_ACTIVE_RECORDS = 2
STAGE1_CLOSE_INACTIVE_RECORDS = 3
STAGE3_REFERENCE_CANDIDATE_STEP_FRAMES = 2
STRONG_CONTINUITY_GAP_FRAMES = 10
MAX_UNION_GAP_FRAMES = FRAME_RATE
MAX_SPATIAL_DISTANCE_CELLS = 2
PER_POINT_MOVEMENT_EVIDENCE_THRESHOLD = 0.50
MOVEMENT_STRENGTH_CHANGE_WEIGHT = 0.40
MOVEMENT_STRENGTH_SPATIAL_WEIGHT = 0.30
MOVEMENT_STRENGTH_TEMPORAL_WEIGHT = 0.30
STAGE3_REFERENCE_WINDOW_FRAMES = 10
STAGE3_ART_STATE_BEFORE_OFFSET_FRAMES = 5
STAGE3_ART_STATE_BEFORE_WINDOW_FRAMES = 10
STAGE3_ART_STATE_AFTER_DELAY_FRAMES = 5
STAGE3_ART_STATE_AFTER_WINDOW_FRAMES = 10
STAGE3_ART_STATE_MIN_SAMPLES = 2
STAGE3_ART_STATE_SIMPLE_SEARCH_FRAMES = FRAME_RATE * 10
STAGE3_ART_STATE_RESCUE_SEARCH_FRAMES = FRAME_RATE * 30
STAGE3_ART_STATE_SEARCH_FRAMES = FRAME_RATE * 3
STAGE3_ART_STATE_SEARCH_MARGIN_FRAMES = 3
STAGE3_ART_STATE_LOCAL_WINDOW_DISTANCE_FRAMES = FRAME_RATE
STAGE3_ART_STATE_MAX_WINDOW_ACTIVITY = 0.12
STAGE3_ART_STATE_MAX_INTERNAL_INSTABILITY = 0.08
STAGE3_ART_STATE_REVEAL_DELAY_FRAMES = 3
STAGE3_ART_STATE_REVEAL_WINDOW_FRAMES = 10
STAGE3_ART_STATE_MIN_REVEAL_HOLD_SCORE = 0.45
STAGE3_ART_STATE_MISSING_REVEAL_SCORE = 0.0
STAGE3_ART_STATE_MIN_FOOTPRINT_SUPPORT_SCORE = 0.10
STAGE3_ART_STATE_FALLBACK_REFERENCE_MIN_FOOTPRINT = 20
STAGE3_FULL_AFTER_FAST_PATH_MAX_FOOTPRINT = 20
STAGE3_LOCAL_CHANGED_CLUSTER_MIN_SIZE = 4
STAGE3_LOCAL_CHANGED_CLUSTER_MIN_COVERAGE = 0.04
STAGE3_LOCAL_CHANGED_CLUSTER_MAX_AMBIGUOUS_COVERAGE = 0.35
STAGE3_MIN_REFERENCE_RECORDS = 2
STAGE3_SURVIVING_THRESHOLD = 0.40
STAGE3_MAX_REFERENCE_ACTIVITY = 0.15
STAGE3_MIN_CONTRAST_SCORE = 0.05
STAGE4_VALID_EVIDENCE_SCORE = 0.70
STAGE4_INVALID_EVIDENCE_SCORE = 0.30
STAGE4_MAX_REFERENCE_ACTIVITY = 0.18
STAGE4_MIN_CONTRAST_SCORE = 0.05
STAGE4_MIN_SUPPORTED_SUBREGION_FOOTPRINT_RATIO = 0.50
STAGE4_MAX_UNRESOLVED_SUBREGION_FOOTPRINT_RATIO_FOR_VALID = 0.20
STAGE4_CORE_ACTIVE_CELL_MIN_TOUCH_COUNT = 2
STAGE4_CORE_ACTIVE_CELL_MIN_ACTIVE_RECORDS = 2
STAGE4_CELL_TIMING_MERGE_TOLERANCE_FRAMES = FRAME_RATE
STAGE4_PROBE_INTERVAL_FRAMES = FRAME_RATE * 2
STAGE4_PROBE_LOCAL_WINDOW_FRAMES = 10
STAGE4_PROBE_HALF_WINDOW_FRAMES = STAGE4_PROBE_LOCAL_WINDOW_FRAMES // 2
STAGE4_PROBE_TERMINAL_OFFSET_FRAMES = 5
STAGE4_PROBE_MIN_ANCHOR_SEPARATION_FRAMES = FRAME_RATE
STAGE4_POST_UNION_LATE_RESOLUTION_FRAMES = 5
STAGE4_PROBE_MIN_CHANGED_SUPPORT = 0.08
STAGE4_PROBE_MIN_JUDGEABLE_CHANGED_SUPPORT = 0.10
STAGE4_PROBE_MIN_ABSOLUTE_CHANGED_CELLS = 1
STAGE4_PROBE_NEGATIVE_MIN_JUDGEABLE_COVERAGE = 0.90
STAGE4_PROBE_NEGATIVE_MAX_UNRESOLVED_SUPPORT = 0.10
STAGE4_PROBE_HOLDING_MIN_UNRESOLVED_SUPPORT = 0.15
STAGE4_OPENING_REVALIDATION_MAX_CHAPTER_OFFSET_FRAMES = FRAME_RATE * 10
STAGE4_OPENING_REVALIDATION_MAX_CHANGED_CELLS = 4
STAGE4_OPENING_REVALIDATION_MAX_CHANGED_SUPPORT = 0.15
STAGE4_OPENING_REVALIDATION_LOOKAHEAD_PROBES = 3
STAGE4_OPENING_REVALIDATION_MIN_UNCHANGED_MATCH_RATIO = 0.75
STAGE4_PROBE_CURRENT_SEARCH_FRAMES = FRAME_RATE * 2
STAGE4_PROBE_LATE_RESOLUTION_MAX_CHANGED_TOUCH_FRAMES = 1
STAGE4_PROBE_LATE_RESOLUTION_MAX_CHANGED_TOUCH_RATIO = 0.20
STAGE4_PROBE_LATE_RESOLUTION_MAX_AFTER_ACTIVITY = 0.05
STAGE4_STATE_RESET_MIN_CHANGED_SUPPORT = 0.55
STAGE4_STATE_RESET_MIN_CHANGED_CELLS = 4
STAGE4_STATE_RESET_MIN_CHANGED_CLUSTER_COUNT = 2
STAGE4_STATE_RESET_MIN_ROW_SPAN = 2
STAGE4_STATE_RESET_MIN_COLUMN_SPAN = 2
STAGE4_POST_RESET_LOCAL_SEARCH_FRAMES = STAGE4_PROBE_LOCAL_WINDOW_FRAMES
STAGE4_NEAR_GLOBAL_OPENING_MIN_CHANGED_SUPPORT = 0.90
STAGE4_NEAR_GLOBAL_OPENING_MAX_UNRESOLVED_CELLS = 9
STAGE4_NEAR_GLOBAL_OPENING_MAX_UNCHANGED_CELLS = 9
STAGE4_NEAR_GLOBAL_OPENING_MIN_TOUCHED_COVERAGE = 0.90
STAGE4_CHANGED_FRONTIER_OPENING_MIN_TOUCHED_COVERAGE = 0.50
STAGE4_CHANGED_FRONTIER_OPENING_MIN_TOUCHED_CELLS = 2
STRONG_ACTIVE_REFERENCE_UNDETERMINED_FLOOR = 0.60
ROCKY_MINIMUM_SIZE_EVIDENCE_FLOOR = 0.60
ACTIVE_REFERENCE_ROCKY_RESCUE_FLOOR = 0.55
LONG_STRONG_UNION_MIN_FRAMES = FRAME_RATE * 20
LONG_STRONG_UNION_STAGE3_SCORE_FLOOR = 0.70
LONG_STRONG_UNION_ACTIVE_REFERENCE_UNDETERMINED_FLOOR = 0.58
HIGH_PARENT_ACTIVITY_PARENT_SCORE_FLOOR = 0.75
HIGH_PARENT_ACTIVITY_RESCUE_FLOOR = 0.58
REFERENCE_UNRELIABLE_PARENT_SCORE_FLOOR = 0.78
REFERENCE_UNRELIABLE_RESCUE_FLOOR = 0.60
STRUCTURAL_GAP_PARENT_SCORE_FLOOR = 0.65
STRUCTURAL_GAP_RESCUE_FLOOR = 0.58
ART_STATE_SUPPORTED_UNION_STAGE3_SCORE_FLOOR = 0.62
ART_STATE_SUPPORTED_UNION_FOOTPRINT_SUPPORT_FLOOR = 0.30
ART_STATE_SUPPORTED_UNION_MAX_FOOTPRINT_SUPPORT = 0.90
ART_STATE_SUPPORTED_RESCUE_FLOOR = 0.58
LONG_STRONG_UNION_MINIMUM_RESCUE_FOOTPRINT = 8
STRUCTURAL_GAP_MINIMUM_RESCUE_FOOTPRINT = 7
ART_STATE_SUPPORTED_MINIMUM_RESCUE_FOOTPRINT = 8
STAGE5_VALID_ANCHOR_PROMOTION_MIN_EVIDENCE = 0.60
STAGE5_VALID_ANCHOR_PROMOTION_MIN_FOOTPRINT = 8
STAGE5_VALID_ANCHOR_PROMOTION_MIN_SIBLING_EVIDENCE = 0.55
STAGE5_VALID_ANCHOR_PROMOTION_MIN_SIBLING_OVERLAP = 0.20
STAGE5_MIN_SUBDIVISION_FRAMES = 15
STAGE5_BOUNDARY_SEARCH_FRAMES = FRAME_RATE * 3
STAGE5_MAX_TRIM_FRAMES = (FRAME_RATE * 2) + (FRAME_RATE // 2)
STAGE5_CONFIRMATION_WINDOW_FRAMES = 3
STAGE5_SKIP_TRIM_NEAR_EDGE_FRAMES = 5
STAGE5_MAX_CANDIDATE_CELLS = 48
STAGE5_BROAD_MOVEMENT_MIN_CELLS = 12
STAGE5_BROAD_MOVEMENT_MIN_FOOTPRINT_RATIO = 0.50
STAGE5_BROAD_MOVEMENT_MIN_CONSECUTIVE_FRAMES = 3
STAGE6_MIN_ROCKY_CLUSTER_FRAMES = FRAME_RATE
STAGE6_MIN_ROCKY_SLICE_COUNT = 4
STAGE6_MIN_ROCKY_CLUSTER_AVERAGE_EVIDENCE = 0.68
STAGE6_MIN_SHORT_ROCKY_SLICE_COUNT = 3
STAGE6_STRONG_QUIET_PARENT_CLUSTER_MIN_FRAMES = FRAME_RATE * 3
STAGE6_STRONG_QUIET_PARENT_CLUSTER_MIN_AVERAGE_EVIDENCE = 0.65
STAGE6_STRONG_QUIET_PARENT_CLUSTER_MIN_FOOTPRINT = 40
STAGE6_STRONG_QUIET_PARENT_MAX_EDGE_ACTIVITY = 0.16
STAGE6_SAME_UNION_MERGE_GAP_FRAMES = FRAME_RATE + (FRAME_RATE // 2)
STAGE6_EXTENDED_GAP_BUDGET_FRAMES = STAGE6_SAME_UNION_MERGE_GAP_FRAMES + 1
STAGE6_ABRUPT_CANVAS_CHANGE_MAX_FRAMES = FRAME_RATE
STAGE6_ABRUPT_CANVAS_CHANGE_MAX_SLICE_COUNT = 2
STAGE6_ABRUPT_CANVAS_CHANGE_MIN_EVIDENCE = 0.64
STAGE6_ABRUPT_CANVAS_CHANGE_MIN_FOOTPRINT = TOTAL_GRID_BLOCKS
STAGE6_ART_STATE_SUPPORTED_CLUSTER_MIN_FRAMES = FRAME_RATE
STAGE6_ART_STATE_SUPPORTED_MIN_SLICE_COUNT = 3
STAGE6_ART_STATE_SUPPORTED_MIN_RESCUE_SLICE_COUNT = 2
STAGE6_ART_STATE_SUPPORTED_MIN_AVERAGE_EVIDENCE = 0.59
STAGE6_VALID_MERGE_MIN_FOOTPRINT_OVERLAP = 0.35
MAX_AUTOMATIC_REUSED_STAGE3_SAMPLE_CACHE_BYTES = 8 * 1024 * 1024 * 1024
ENABLE_REUSABLE_STAGE3_SAMPLE_CACHE_WRITES = False


@dataclass(frozen=True)
class MovementEvidenceRecord:
    record_index: int
    evaluation_point_timecode: str
    frame_index: int
    movement_present: bool
    touched_grid_coordinates: tuple[GridCoordinate, ...]
    touched_grid_coordinate_count: int
    change_magnitude_score: float
    spatial_extent_score: float
    temporal_persistence_score: float
    movement_strength_score: float
    opening_signal: bool = False
    continuation_signal: bool = False
    weak_signal: bool = False


@dataclass(frozen=True)
class MovementSpan:
    span_index: int
    start_frame: int
    end_frame: int
    start_time: str
    end_time: str
    footprint: frozenset[GridCoordinate]
    footprint_size: int
    record_indices: tuple[int, ...]


@dataclass(frozen=True)
class CandidateUnion:
    union_index: int
    start_frame: int
    end_frame: int
    start_time: str
    end_time: str
    member_movement_spans: tuple[MovementSpan, ...]
    union_footprint: frozenset[GridCoordinate]
    union_footprint_size: int


@dataclass(frozen=True)
class ScreenedCandidateUnion:
    candidate_union: CandidateUnion
    screening_result: str
    surviving: bool
    provisional_survival: bool
    reason: str
    within_union_record_count: int
    before_record_count: int
    after_record_count: int
    mean_movement_strength: float
    mean_temporal_persistence: float
    mean_spatial_extent: float
    lasting_change_evidence_score: float
    before_reference_activity: float
    after_reference_activity: float
    reference_windows_reliable: bool
    stage3_mode: str = 'art_state'
    stage3_alignment_mode: str = 'none'
    stage3_persistent_difference_score: float = 0.0
    stage3_footprint_support_score: float = 0.0
    stage3_after_window_persistence_score: float = 0.0
    stage3_before_window_start: str | None = None
    stage3_before_window_end: str | None = None
    stage3_after_window_start: str | None = None
    stage3_after_window_end: str | None = None
    stage3_before_sample_count: int = 0
    stage3_after_sample_count: int = 0
    stage3_reveal_sample_count: int = 0
    stage3_before_window_quality_score: float = 0.0
    stage3_after_window_quality_score: float = 0.0
    stage3_reveal_window_quality_score: float = 0.0
    stage3_before_window_candidate_count: int = 0
    stage3_after_window_candidate_count: int = 0
    stage3_reveal_window_candidate_count: int = 0
    stage3_before_window_tier: str | None = None
    stage3_after_window_tier: str | None = None
    stage3_reveal_window_tier: str | None = None
    stage3_reveal_window_start: str | None = None
    stage3_reveal_window_end: str | None = None
    stage3_reveal_window_hold_score: float = 0.0
    stage3_debug_trace: dict[str, object] | None = None


@dataclass(frozen=True)
class ClassifiedTimeSlice:
    slice_index: int
    parent_union_index: int
    slice_level: int
    start_frame: int
    end_frame: int
    start_time: str
    end_time: str
    footprint: frozenset[GridCoordinate]
    footprint_size: int
    within_slice_record_count: int
    classification: str
    reason: str
    lasting_change_evidence_score: float
    before_reference_activity: float
    after_reference_activity: float
    reference_windows_reliable: bool
    parent_range: tuple[int, int] | None = None
    stage5_debug: dict[str, object] | None = None

@dataclass(frozen=True)
class FinalCandidateRange:
    range_index: int
    parent_union_index: int
    start_frame: int
    end_frame: int
    start_time: str
    end_time: str
    source_classifications: tuple[str, ...]
    includes_boundary: bool
    boundary_count: int



# ============================================================
# SECTION C - Record Scoring And Grid Helpers
# ============================================================


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))



def extract_canvas_region(frame, cv2):
    frame_height, frame_width = frame.shape[:2]
    left = int(frame_width * CANVAS_LEFT_RATIO)
    right = int(frame_width * CANVAS_RIGHT_RATIO)
    top = int(frame_height * CANVAS_TOP_RATIO)
    bottom = int(frame_height * CANVAS_BOTTOM_RATIO)
    canvas_frame = frame[top:bottom, left:right]
    gray = cv2.cvtColor(canvas_frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def get_stage3_sample_gray(sample: dict[str, object]):
    gray = sample.get('gray')
    if gray is not None:
        return gray
    return sample['art_gray']



def extract_touched_grid_coordinates(mask) -> tuple[GridCoordinate, ...]:
    grid_height = mask.shape[0] // GRID_ROWS
    grid_width = mask.shape[1] // GRID_COLUMNS
    if grid_height <= 0 or grid_width <= 0:
        return ()

    trimmed = mask[: grid_height * GRID_ROWS, : grid_width * GRID_COLUMNS]
    block_grid = trimmed.reshape(GRID_ROWS, grid_height, GRID_COLUMNS, grid_width)
    block_activity = block_grid.mean(axis=(1, 3)) / 255.0
    touched: list[GridCoordinate] = []
    for row_index in range(GRID_ROWS):
        for column_index in range(GRID_COLUMNS):
            if block_activity[row_index, column_index] > 0:
                touched.append((row_index, column_index))
    return tuple(touched)


def compute_movement_strength_score(
    change_magnitude_score: float,
    spatial_extent_score: float,
    temporal_persistence_score: float,
) -> float:
    return clamp_score(
        (MOVEMENT_STRENGTH_CHANGE_WEIGHT * change_magnitude_score)
        + (MOVEMENT_STRENGTH_SPATIAL_WEIGHT * spatial_extent_score)
        + (MOVEMENT_STRENGTH_TEMPORAL_WEIGHT * temporal_persistence_score)
    )




def build_movement_evidence_record(
    record_index: int,
    previous_sample: dict[str, object],
    current_sample: dict[str, object],
    next_sample: dict[str, object],
    recent_persistent_masks: deque,
    settings: DetectorSettings,
    cv2,
) -> MovementEvidenceRecord:
    adjacent_mask, persistent_mask = build_persistent_change_mask(
        previous_gray=previous_sample['gray'],
        current_gray=current_sample['gray'],
        next_gray=next_sample['gray'],
        settings=settings,
        cv2=cv2,
    )
    recent_persistent_masks.append(persistent_mask.copy())
    trail_mask = build_trail_mask(recent_persistent_masks, cv2)
    adjacent_change_score = float(adjacent_mask.mean()) / 255.0
    persistent_change_score = float(persistent_mask.mean()) / 255.0
    adjacent_blocks = count_active_blocks(adjacent_mask)
    persistent_blocks = count_active_blocks(persistent_mask)
    trail_blocks = count_active_blocks(trail_mask) if trail_mask is not None else 0
    trail_excess_blocks = max(0, trail_blocks - persistent_blocks)
    touched_grid_coordinates = extract_touched_grid_coordinates(persistent_mask)
    touched_grid_coordinate_count = len(touched_grid_coordinates)

    weak_signal = is_weak_art_change_signal(
        adjacent_ratio=adjacent_change_score,
        persistent_ratio=persistent_change_score,
        adjacent_blocks=adjacent_blocks,
        persistent_blocks=persistent_blocks,
        settings=settings,
    )
    opening_signal = should_enter_active_state(
        adjacent_ratio=adjacent_change_score,
        persistent_ratio=persistent_change_score,
        adjacent_blocks=adjacent_blocks,
        persistent_blocks=persistent_blocks,
        trail_excess_blocks=trail_excess_blocks,
        settings=settings,
    )
    continuation_signal = should_remain_active_state(
        adjacent_ratio=adjacent_change_score,
        persistent_ratio=persistent_change_score,
        adjacent_blocks=adjacent_blocks,
        persistent_blocks=persistent_blocks,
        trail_excess_blocks=trail_excess_blocks,
        settings=settings,
    )
    change_magnitude_score = clamp_score(adjacent_change_score / max(settings.active_pixel_ratio, 1e-6))
    spatial_extent_score = clamp_score(touched_grid_coordinate_count / max(1, MAX_ACTIVE_BLOCKS))
    temporal_persistence_score = clamp_score(persistent_change_score / max(settings.active_pixel_ratio * 0.5, 1e-6))
    movement_strength_score = compute_movement_strength_score(
        change_magnitude_score=change_magnitude_score,
        spatial_extent_score=spatial_extent_score,
        temporal_persistence_score=temporal_persistence_score,
    )

    return MovementEvidenceRecord(
        record_index=record_index,
        evaluation_point_timecode=Timecode(total_frames=int(current_sample['frame_index'])).to_hhmmssff(),
        frame_index=int(current_sample['frame_index']),
        movement_present=bool(
            movement_strength_score >= PER_POINT_MOVEMENT_EVIDENCE_THRESHOLD
            and touched_grid_coordinate_count >= 1
        ),
        touched_grid_coordinates=touched_grid_coordinates,
        touched_grid_coordinate_count=touched_grid_coordinate_count,
        change_magnitude_score=change_magnitude_score,
        spatial_extent_score=spatial_extent_score,
        temporal_persistence_score=temporal_persistence_score,
        movement_strength_score=movement_strength_score,
        opening_signal=opening_signal,
        continuation_signal=continuation_signal,
        weak_signal=weak_signal,
    )


# ============================================================
# SECTION D - Stage 1 Movement Evidence And Movement Spans
# ============================================================


def detect_movement_evidence_records(
    video_path: Path,
    chapter_range: ChapterRange,
    settings: DetectorSettings,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[list[MovementEvidenceRecord], list[dict[str, object]]]:
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('OpenCV is required for staged movement detection.') from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f'Unable to open video file: {video_path}')

    recent_persistent_masks: deque = deque(maxlen=TRAIL_MASK_WINDOW)
    sampled_frames: deque[dict[str, object]] = deque(maxlen=3)
    retained_stage3_samples: list[dict[str, object]] = []
    records: list[MovementEvidenceRecord] = []
    last_progress_percent = -PROGRESS_STEP_PERCENT
    record_index = 1

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
                canvas_gray = extract_canvas_region(frame, cv2)
                stage1_sample = {
                    'frame_index': current_frame,
                    'gray': canvas_gray,
                }
                sampled_frames.append(stage1_sample)
                retained_stage3_samples.append(
                    {
                        'frame_index': current_frame,
                        'canvas_shape': tuple(int(dimension) for dimension in canvas_gray.shape),
                        'gray': canvas_gray,
                    }
                )
                if len(sampled_frames) == 3:
                    records.append(
                        build_movement_evidence_record(
                            record_index=record_index,
                            previous_sample=sampled_frames[0],
                            current_sample=sampled_frames[1],
                            next_sample=sampled_frames[2],
                            recent_persistent_masks=recent_persistent_masks,
                            settings=settings,
                            cv2=cv2,
                        )
                    )
                    record_index += 1
            current_frame += 1

        if len(sampled_frames) >= 2:
            records.append(
                build_movement_evidence_record(
                    record_index=record_index,
                    previous_sample=sampled_frames[-2],
                    current_sample=sampled_frames[-1],
                    next_sample=sampled_frames[-1],
                    recent_persistent_masks=recent_persistent_masks,
                    settings=settings,
                    cv2=cv2,
                )
            )
    finally:
        capture.release()

    return records, retained_stage3_samples



def infer_stage1_record_frame_step(
    records: list[MovementEvidenceRecord],
    fallback_step: int,
) -> int:
    frame_gaps = [
        current_record.frame_index - previous_record.frame_index
        for previous_record, current_record in zip(records, records[1:])
        if current_record.frame_index > previous_record.frame_index
    ]
    if not frame_gaps:
        return max(1, fallback_step)
    return min(frame_gaps)



def stage1_opening_window(
    records: list[MovementEvidenceRecord],
    current_index: int,
) -> list[MovementEvidenceRecord]:
    window_start = max(0, current_index - (STAGE1_OPEN_WINDOW_RECORDS - 1))
    return records[window_start: current_index + 1]



def build_stage1_movement_spans(
    records: Iterable[MovementEvidenceRecord],
    settings: DetectorSettings,
) -> list[MovementSpan]:
    ordered_records = list(records)
    if not ordered_records:
        return []

    inferred_frame_step = infer_stage1_record_frame_step(ordered_records, settings.sample_stride)
    spans: list[MovementSpan] = []
    active_start_frame: int | None = None
    active_record_indices: list[int] = []
    active_footprint: set[GridCoordinate] = set()
    inactive_streak = 0
    inactive_streak_start_frame: int | None = None
    last_active_frame: int | None = None

    def close_span(end_frame: int) -> None:
        nonlocal active_start_frame, active_record_indices, active_footprint, inactive_streak
        nonlocal inactive_streak_start_frame, last_active_frame
        if active_start_frame is None or not active_record_indices:
            return
        spans.append(
            MovementSpan(
                span_index=len(spans) + 1,
                start_frame=active_start_frame,
                end_frame=end_frame,
                start_time=Timecode(total_frames=active_start_frame).to_hhmmssff(),
                end_time=Timecode(total_frames=end_frame).to_hhmmssff(),
                footprint=frozenset(active_footprint),
                footprint_size=len(active_footprint),
                record_indices=tuple(active_record_indices),
            )
        )
        active_start_frame = None
        active_record_indices = []
        active_footprint = set()
        inactive_streak = 0
        inactive_streak_start_frame = None
        last_active_frame = None

    for current_index, record in enumerate(ordered_records):
        if active_start_frame is None:
            opening_window = stage1_opening_window(ordered_records, current_index)
            active_window_records = [
                window_record for window_record in opening_window if window_record.movement_present
            ]
            if len(active_window_records) >= STAGE1_OPEN_MIN_ACTIVE_RECORDS:
                active_start_frame = active_window_records[0].frame_index
                active_record_indices = []
                active_footprint = set()
                for window_record in active_window_records:
                    if window_record.record_index not in active_record_indices:
                        active_record_indices.append(window_record.record_index)
                    active_footprint.update(window_record.touched_grid_coordinates)
                last_active_frame = active_window_records[-1].frame_index
                inactive_streak = 0
                inactive_streak_start_frame = None
            else:
                continue

        if record.movement_present:
            inactive_streak = 0
            inactive_streak_start_frame = None
            last_active_frame = record.frame_index
            if record.record_index not in active_record_indices:
                active_record_indices.append(record.record_index)
            active_footprint.update(record.touched_grid_coordinates)
        else:
            inactive_streak += 1
            if inactive_streak_start_frame is None:
                inactive_streak_start_frame = record.frame_index
            if inactive_streak >= STAGE1_CLOSE_INACTIVE_RECORDS:
                close_span(inactive_streak_start_frame)

    if active_start_frame is not None and active_record_indices and last_active_frame is not None:
        close_span(last_active_frame + inferred_frame_step)

    minimum_frames = settings.min_burst_length.total_frames
    return [span for span in spans if (span.end_frame - span.start_frame) >= minimum_frames]
# ============================================================
# SECTION E - Stage 2 Candidate Union Construction
# ============================================================


MAX_STAGE2_WEAK_PATH_NEW_CELL_GROWTH = 4
MIN_STAGE2_WEAK_PATH_ATTACHMENT_RATIO = 0.50



def footprints_overlap(first: frozenset[GridCoordinate], second: frozenset[GridCoordinate]) -> bool:
    return any(coordinate in second for coordinate in first)



def footprints_adjacent(first: frozenset[GridCoordinate], second: frozenset[GridCoordinate]) -> bool:
    for first_row, first_column in first:
        for second_row, second_column in second:
            if max(abs(first_row - second_row), abs(first_column - second_column)) <= 1:
                return True
    return False



def compute_spatial_distance(first: frozenset[GridCoordinate], second: frozenset[GridCoordinate]) -> int:
    if not first or not second:
        return GRID_ROWS + GRID_COLUMNS
    return min(
        max(abs(first_row - second_row), abs(first_column - second_column))
        for first_row, first_column in first
        for second_row, second_column in second
    )



def span_has_spatial_support(
    next_footprint: frozenset[GridCoordinate],
    comparison_footprint: frozenset[GridCoordinate],
) -> bool:
    return (
        footprints_overlap(next_footprint, comparison_footprint)
        or footprints_adjacent(next_footprint, comparison_footprint)
        or compute_spatial_distance(next_footprint, comparison_footprint) <= MAX_SPATIAL_DISTANCE_CELLS
    )



def compute_stage2_new_cells(
    current_union_footprint: frozenset[GridCoordinate],
    next_span_footprint: frozenset[GridCoordinate],
) -> frozenset[GridCoordinate]:
    return frozenset(
        coordinate for coordinate in next_span_footprint
        if coordinate not in current_union_footprint
    )



def compute_stage2_new_cell_attachment_ratio(
    current_union_footprint: frozenset[GridCoordinate],
    new_cells: frozenset[GridCoordinate],
) -> float:
    if not new_cells:
        return 1.0

    attached_cells = sum(
        1
        for coordinate in new_cells
        if span_has_spatial_support(frozenset({coordinate}), current_union_footprint)
    )
    return attached_cells / len(new_cells)



def merge_causes_large_stage2_expansion(
    current_union_footprint: frozenset[GridCoordinate],
    next_span_footprint: frozenset[GridCoordinate],
) -> bool:
    if not current_union_footprint:
        return False

    new_cells = compute_stage2_new_cells(current_union_footprint, next_span_footprint)
    attachment_ratio = compute_stage2_new_cell_attachment_ratio(current_union_footprint, new_cells)
    return (
        len(new_cells) > MAX_STAGE2_WEAK_PATH_NEW_CELL_GROWTH
        and attachment_ratio < MIN_STAGE2_WEAK_PATH_ATTACHMENT_RATIO
    )



def build_stage2_candidate_unions(
    movement_spans: Iterable[MovementSpan],
) -> list[CandidateUnion]:
    ordered_spans = sorted(movement_spans, key=lambda span: span.start_frame)
    if not ordered_spans:
        return []

    unions: list[CandidateUnion] = []
    current_spans: list[MovementSpan] = [ordered_spans[0]]
    current_footprint: set[GridCoordinate] = set(ordered_spans[0].footprint)

    def close_union(union_index: int) -> None:
        union_start = current_spans[0].start_frame
        union_end = current_spans[-1].end_frame
        unions.append(
            CandidateUnion(
                union_index=union_index,
                start_frame=union_start,
                end_frame=union_end,
                start_time=Timecode(total_frames=union_start).to_hhmmssff(),
                end_time=Timecode(total_frames=union_end).to_hhmmssff(),
                member_movement_spans=tuple(current_spans),
                union_footprint=frozenset(current_footprint),
                union_footprint_size=len(current_footprint),
            )
        )

    for next_span in ordered_spans[1:]:
        previous_span = current_spans[-1]
        temporal_gap = next_span.start_frame - previous_span.end_frame
        union_footprint = frozenset(current_footprint)
        union_support = span_has_spatial_support(next_span.footprint, union_footprint)
        weak_path_blocked_by_guardrail = merge_causes_large_stage2_expansion(union_footprint, next_span.footprint)
        should_merge = False
        if temporal_gap <= STRONG_CONTINUITY_GAP_FRAMES:
            should_merge = True
        elif temporal_gap <= MAX_UNION_GAP_FRAMES and union_support and not weak_path_blocked_by_guardrail:
            should_merge = True

        if should_merge:
            current_spans.append(next_span)
            current_footprint.update(next_span.footprint)
        else:
            close_union(len(unions) + 1)
            current_spans = [next_span]
            current_footprint = set(next_span.footprint)

    close_union(len(unions) + 1)
    return unions
# ============================================================
# SECTION F - Stage 3 Candidate Union Screening
# ============================================================


def collect_records_in_frame_range(
    records: Iterable[MovementEvidenceRecord],
    start_frame: int,
    end_frame: int,
) -> list[MovementEvidenceRecord]:
    return [
        record
        for record in records
        if start_frame <= record.frame_index < end_frame
    ]




def build_stage3_runtime_context(
    ordered_records: list[MovementEvidenceRecord],
    sampled_frames: list[dict[str, object]],
) -> dict[str, object]:
    return {
        'ordered_records': ordered_records,
        'record_frame_indices': [record.frame_index for record in ordered_records],
        'sampled_frames': sampled_frames,
        'sample_frame_indices': [int(sample['frame_index']) for sample in sampled_frames],
        'record_range_cache': {},
        'sample_range_cache': {},
        'window_metadata_cache': {},
        'window_active_coordinate_cache': {},
        'cell_movement_cache': {},
        'outside_footprint_cache': {},
        'outside_coordinate_cache': {},
        'cell_trust_cache': {},
        'baseline_cache': {},
    }


def get_stage3_window_records(
    stage3_context: dict[str, object] | None,
    ordered_records: list[MovementEvidenceRecord],
    window_start: int,
    window_end: int,
) -> list[MovementEvidenceRecord]:
    if stage3_context is None:
        return collect_records_in_frame_range(ordered_records, window_start, window_end)
    cache = stage3_context['record_range_cache']
    cache_key = (window_start, window_end)
    if cache_key not in cache:
        frame_indices = stage3_context['record_frame_indices']
        start_index = bisect_left(frame_indices, window_start)
        end_index = bisect_left(frame_indices, window_end)
        cache[cache_key] = ordered_records[start_index:end_index]
    return cache[cache_key]


def get_stage3_window_samples(
    stage3_context: dict[str, object] | None,
    sampled_frames: list[dict[str, object]],
    window_start: int,
    window_end: int,
) -> list[dict[str, object]]:
    if stage3_context is None:
        return collect_window_samples(sampled_frames, window_start, window_end)
    cache = stage3_context['sample_range_cache']
    cache_key = (window_start, window_end)
    if cache_key not in cache:
        frame_indices = stage3_context['sample_frame_indices']
        start_index = bisect_left(frame_indices, window_start)
        end_index = bisect_left(frame_indices, window_end)
        cache[cache_key] = sampled_frames[start_index:end_index]
    return cache[cache_key]

def get_stage3_window_active_coordinates(
    stage3_context: dict[str, object] | None,
    ordered_records: list[MovementEvidenceRecord],
    window_start: int,
    window_end: int,
) -> frozenset[GridCoordinate]:
    if stage3_context is None:
        return frozenset(
            coordinate
            for record in ordered_records
            if window_start <= record.frame_index < window_end and record.movement_present
            for coordinate in record.touched_grid_coordinates
        )
    cache = stage3_context['window_active_coordinate_cache']
    cache_key = (window_start, window_end)
    if cache_key not in cache:
        cache[cache_key] = frozenset(
            coordinate
            for record in get_stage3_window_records(stage3_context, ordered_records, window_start, window_end)
            if record.movement_present
            for coordinate in record.touched_grid_coordinates
        )
    return cache[cache_key]

def average_record_score(records: Iterable[MovementEvidenceRecord], attribute_name: str) -> float:
    selected_records = list(records)
    if not selected_records:
        return 0.0
    return sum(float(getattr(record, attribute_name)) for record in selected_records) / len(selected_records)



def build_footprint_from_records(records: Iterable[MovementEvidenceRecord]) -> frozenset[GridCoordinate]:
    footprint: set[GridCoordinate] = set()
    for record in records:
        footprint.update(record.touched_grid_coordinates)
    return frozenset(footprint)



def collect_stage3_art_state_samples(
    video_path: Path,
    chapter_range: ChapterRange,
    settings: DetectorSettings,
    status_callback: Callable[[str], None] | None = None,
    precomputed_samples: list[dict[str, object]] | None = None,
    reuse_source_description: str | None = None,
) -> list[dict[str, object]]:
    total_frames = max(0, chapter_range.end.total_frames - chapter_range.start.total_frames)
    expected_sample_count = 0 if total_frames <= 0 else ((total_frames - 1) // max(1, settings.sample_stride)) + 1
    progress_interval = max(1, expected_sample_count // 20) if expected_sample_count > 0 else 1
    last_reported_sample_count = 0
    stage_started_at = time.perf_counter()

    if precomputed_samples is not None:
        sampled_frames = list(precomputed_samples)
        for sampled_frame_count in range(1, len(sampled_frames) + 1):
            should_report_progress = (
                status_callback is not None
                and (
                    sampled_frame_count == 1
                    or sampled_frame_count == len(sampled_frames)
                    or (sampled_frame_count - last_reported_sample_count) >= progress_interval
                )
            )
            if should_report_progress:
                last_reported_sample_count = sampled_frame_count
                elapsed = format_stage_elapsed(time.perf_counter() - stage_started_at)
                status_callback(
                    f"Runtime Stage 2B - Collecting Stage 3 art-state samples: "
                    f"{sampled_frame_count}/{max(1, expected_sample_count)} sampled frames "
                    f"(elapsed {elapsed}; {reuse_source_description or 'reusing precomputed Stage 3 art-state samples'})"
                )
        return sampled_frames

    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('OpenCV is required for staged Stage 3 art-state screening.') from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f'Unable to open video file: {video_path}')

    sampled_frames: list[dict[str, object]] = []
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, chapter_range.start.total_frames)
        current_frame = chapter_range.start.total_frames
        while current_frame < chapter_range.end.total_frames:
            success, frame = capture.read()
            if not success:
                break
            if ((current_frame - chapter_range.start.total_frames) % settings.sample_stride) == 0:
                canvas_gray = extract_canvas_region(frame, cv2)
                sampled_frames.append(
                    {
                        'frame_index': current_frame,
                        'canvas_shape': tuple(int(dimension) for dimension in canvas_gray.shape),
                        'gray': canvas_gray,
                    }
                )
                sampled_frame_count = len(sampled_frames)
                should_report_progress = (
                    status_callback is not None
                    and (
                        sampled_frame_count == 1
                        or sampled_frame_count == expected_sample_count
                        or (sampled_frame_count - last_reported_sample_count) >= progress_interval
                    )
                )
                if should_report_progress:
                    last_reported_sample_count = sampled_frame_count
                    elapsed = format_stage_elapsed(time.perf_counter() - stage_started_at)
                    status_callback(
                        f"Runtime Stage 2B - Collecting Stage 3 art-state samples: "
                        f"{sampled_frame_count}/{expected_sample_count} sampled frames "
                        f"(elapsed {elapsed})"
                    )
            current_frame += 1
    finally:
        capture.release()

    return sampled_frames



def compute_persistent_difference_score(records: Iterable[MovementEvidenceRecord]) -> float:
    mean_movement_strength = average_record_score(records, 'movement_strength_score')
    mean_temporal_persistence = average_record_score(records, 'temporal_persistence_score')
    return clamp_score((mean_movement_strength + mean_temporal_persistence) / 2.0)



def compute_footprint_support_score(
    records: Iterable[MovementEvidenceRecord],
    footprint_size: int,
    footprint_scale: float,
) -> float:
    mean_spatial_extent = average_record_score(records, 'spatial_extent_score')
    footprint_extent_score = clamp_score(footprint_size / max(1, TOTAL_GRID_BLOCKS * footprint_scale))
    return clamp_score((mean_spatial_extent + footprint_extent_score) / 2.0)



def compute_after_window_persistence_score(after_reference_activity: float) -> float:
    return clamp_score(1.0 - after_reference_activity)



def build_stage3_art_state_search_ranges(
    candidate_union: CandidateUnion,
    chapter_range: ChapterRange,
) -> tuple[tuple[int, int], tuple[int, int]]:
    before_search_start = max(
        chapter_range.start.total_frames,
        candidate_union.start_frame - STAGE3_ART_STATE_SEARCH_FRAMES,
    )
    before_search_end = max(
        before_search_start,
        candidate_union.start_frame - STAGE3_ART_STATE_SEARCH_MARGIN_FRAMES,
    )
    after_search_start = min(
        chapter_range.end.total_frames,
        candidate_union.end_frame + STAGE3_ART_STATE_SEARCH_MARGIN_FRAMES,
    )
    after_search_end = min(
        chapter_range.end.total_frames,
        candidate_union.end_frame + STAGE3_ART_STATE_SEARCH_FRAMES,
    )
    return (before_search_start, before_search_end), (after_search_start, after_search_end)


def compute_stage3_window_internal_instability(
    window_samples: list[dict[str, object]],
    settings: DetectorSettings,
    cv2,
) -> float:
    if len(window_samples) < 2:
        return 0.0

    instability_ratios: list[float] = []
    for sample_index in range(len(window_samples) - 1):
        instability_mask = build_art_state_change_mask(get_stage3_sample_gray(window_samples[sample_index]), get_stage3_sample_gray(window_samples[sample_index + 1]), settings, cv2)
        instability_ratios.append(compute_binary_mask_ratio(instability_mask))
    if not instability_ratios:
        return 0.0
    return sum(instability_ratios) / len(instability_ratios)


def build_stage3_reference_window_candidates(
    search_start: int,
    search_end: int,
    window_frames: int,
    step_frames: int,
) -> list[tuple[int, int]]:
    if search_end <= search_start:
        return []
    if (search_end - search_start) <= window_frames:
        return [(search_start, search_end)]

    candidates: list[tuple[int, int]] = []
    candidate_start = search_start
    latest_start = search_end - window_frames
    while candidate_start <= latest_start:
        candidates.append((candidate_start, candidate_start + window_frames))
        candidate_start += max(1, step_frames)
    final_candidate = (latest_start, search_end)
    if not candidates or candidates[-1] != final_candidate:
        candidates.append(final_candidate)
    return candidates



def get_stage3_reference_candidate_step_frames(settings: DetectorSettings) -> int:
    return max(STAGE3_REFERENCE_CANDIDATE_STEP_FRAMES, settings.sample_stride)



def score_stage3_reference_window(
    candidate_window: tuple[int, int],
    records: Iterable[MovementEvidenceRecord],
    sampled_frames: list[dict[str, object]],
    union_anchor_frame: int,
    search_radius_frames: int,
    settings: DetectorSettings,
    cv2,
) -> dict[str, object] | None:
    window_start, window_end = candidate_window
    window_samples = collect_window_samples(sampled_frames, window_start, window_end)
    if len(window_samples) < STAGE3_ART_STATE_MIN_SAMPLES:
        return None

    window_records = collect_records_in_frame_range(records, window_start, window_end)
    mean_window_activity = average_record_score(window_records, 'movement_strength_score')
    internal_instability = compute_stage3_window_internal_instability(window_samples, settings, cv2)
    if union_anchor_frame <= window_start:
        distance_frames = max(0, window_start - union_anchor_frame)
    else:
        distance_frames = max(0, union_anchor_frame - window_end)
    activity_score = 1.0 - clamp_score(mean_window_activity / max(STAGE3_ART_STATE_MAX_WINDOW_ACTIVITY, 1e-6))
    instability_score = 1.0 - clamp_score(internal_instability / max(STAGE3_ART_STATE_MAX_INTERNAL_INSTABILITY, 1e-6))
    distance_score = 1.0 - clamp_score(distance_frames / max(1, search_radius_frames))
    quality_score = clamp_score(
        (0.45 * activity_score)
        + (0.45 * instability_score)
        + (0.10 * distance_score)
    )
    return {
        'window_start': window_start,
        'window_end': window_end,
        'sample_count': len(window_samples),
        'mean_window_activity': mean_window_activity,
        'internal_instability': internal_instability,
        'distance_frames': distance_frames,
        'quality_score': quality_score,
        'tier': 'local' if distance_frames <= STAGE3_ART_STATE_LOCAL_WINDOW_DISTANCE_FRAMES else 'fallback',
    }



def select_stage3_art_state_reference_window(
    *,
    search_start: int,
    search_end: int,
    union_anchor_frame: int,
    records: Iterable[MovementEvidenceRecord],
    sampled_frames: list[dict[str, object]],
    settings: DetectorSettings,
    cv2,
    stage3_context: dict[str, object] | None = None,
) -> tuple[dict[str, object] | None, int]:
    candidate_windows = build_stage3_reference_window_candidates(
        search_start,
        search_end,
        STAGE3_ART_STATE_BEFORE_WINDOW_FRAMES,
        get_stage3_reference_candidate_step_frames(settings),
    )
    scored_candidates: list[dict[str, object]] = []
    for candidate_window in candidate_windows:
        scored_candidate = score_stage3_reference_window(
            candidate_window,
            records,
            sampled_frames,
            union_anchor_frame,
            STAGE3_ART_STATE_SEARCH_FRAMES,
            settings,
            cv2,
        )
        if scored_candidate is not None:
            scored_candidates.append(scored_candidate)
    if not scored_candidates:
        return None, len(candidate_windows)
    best_candidate = max(
        scored_candidates,
        key=lambda candidate: (candidate['quality_score'], -candidate['distance_frames']),
    )
    return best_candidate, len(candidate_windows)



def build_stage3_reveal_search_range(
    after_window_end: int,
    chapter_range: ChapterRange,
    next_union_start: int | None,
) -> tuple[int, int]:
    reveal_search_start = min(
        chapter_range.end.total_frames,
        after_window_end + STAGE3_ART_STATE_REVEAL_DELAY_FRAMES,
    )
    reveal_search_end = min(
        chapter_range.end.total_frames,
        reveal_search_start + STAGE3_ART_STATE_SEARCH_FRAMES,
    )
    if next_union_start is not None:
        reveal_search_end = min(reveal_search_end, next_union_start)
    return reveal_search_start, max(reveal_search_start, reveal_search_end)



def build_stage3_art_state_window_baseline(
    sampled_frames: list[dict[str, object]],
    window_start: int,
    window_end: int,
) -> tuple[list[dict[str, object]], object | None]:
    window_samples = collect_window_samples(sampled_frames, window_start, window_end)
    return window_samples, build_median_baseline(window_samples)



def compute_stage3_art_state_reveal_hold_score(
    pre_baseline,
    post_baseline,
    reveal_baseline,
    reveal_samples: list[dict[str, object]],
    art_state_footprint_mask,
    footprint_size: int,
    settings: DetectorSettings,
    cv2,
) -> float:
    if reveal_baseline is None or not reveal_samples:
        return 0.0

    persistent_mask = build_art_state_change_mask(pre_baseline, reveal_baseline, settings, cv2)
    focused_persistent_mask = cv2.bitwise_and(persistent_mask, art_state_footprint_mask)
    persistent_score = compute_stage3_art_state_persistent_difference_score(
        focused_persistent_mask,
        footprint_size,
        settings,
    )

    drift_scores: list[float] = []
    reveal_stability_mask = build_art_state_change_mask(post_baseline, reveal_baseline, settings, cv2)
    focused_reveal_stability_mask = cv2.bitwise_and(reveal_stability_mask, art_state_footprint_mask)
    drift_scores.append(
        clamp_score(compute_binary_mask_ratio(focused_reveal_stability_mask) / max(settings.active_pixel_ratio * 2.0, 1e-6))
    )
    for sample in reveal_samples:
        drift_mask = build_art_state_change_mask(post_baseline, get_stage3_sample_gray(sample), settings, cv2)
        focused_drift_mask = cv2.bitwise_and(drift_mask, art_state_footprint_mask)
        drift_scores.append(
            clamp_score(compute_binary_mask_ratio(focused_drift_mask) / max(settings.active_pixel_ratio * 2.0, 1e-6))
        )

    mean_drift = sum(drift_scores) / len(drift_scores)
    return clamp_score(persistent_score - (mean_drift * 0.75))



def build_union_canvas_footprint_mask(
    candidate_union: CandidateUnion,
    canvas_shape: tuple[int, int],
):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('NumPy is required for staged Stage 3 art-state screening.') from exc

    mask = np.zeros(canvas_shape, dtype=np.uint8)
    canvas_height, canvas_width = canvas_shape
    cell_height = max(1, canvas_height // GRID_ROWS)
    cell_width = max(1, canvas_width // GRID_COLUMNS)

    for row_index, column_index in candidate_union.union_footprint:
        top = row_index * cell_height
        left = column_index * cell_width
        bottom = canvas_height if row_index == (GRID_ROWS - 1) else min(canvas_height, (row_index + 1) * cell_height)
        right = canvas_width if column_index == (GRID_COLUMNS - 1) else min(canvas_width, (column_index + 1) * cell_width)
        mask[top:bottom, left:right] = 255
    return mask





def compute_binary_mask_ratio(mask) -> float:
    if mask is None or getattr(mask, 'size', 0) == 0:
        return 0.0
    return float(mask.mean()) / 255.0



def compute_stage3_art_state_persistent_difference_score(
    focused_change_mask,
    footprint_size: int,
    settings: DetectorSettings,
) -> float:
    focused_ratio = compute_binary_mask_ratio(focused_change_mask)
    focused_blocks = count_active_blocks(focused_change_mask)
    ratio_score = clamp_score(focused_ratio / max(settings.active_pixel_ratio * 4.0, 1e-6))
    block_score = clamp_score(focused_blocks / max(1.0, min(float(footprint_size), 12.0)))
    return clamp_score((ratio_score + block_score) / 2.0)



def compute_stage3_art_state_footprint_support_score(
    focused_change_mask,
    footprint_size: int,
) -> float:
    focused_blocks = count_active_blocks(focused_change_mask)
    return clamp_score(focused_blocks / max(1.0, float(footprint_size)))



def compute_stage3_art_state_after_window_persistence_score(
    pre_baseline,
    post_baseline,
    after_samples: list[dict[str, object]],
    art_state_footprint_mask,
    footprint_size: int,
    settings: DetectorSettings,
    cv2,
) -> float:
    if not after_samples:
        return 0.0

    persistence_scores: list[float] = []
    instability_scores: list[float] = []
    for sample in after_samples:
        persistent_mask = build_art_state_change_mask(pre_baseline, get_stage3_sample_gray(sample), settings, cv2)
        focused_persistent_mask = cv2.bitwise_and(persistent_mask, art_state_footprint_mask)
        persistence_scores.append(
            compute_stage3_art_state_persistent_difference_score(
                focused_persistent_mask,
                footprint_size,
                settings,
            )
        )
        instability_mask = build_art_state_change_mask(post_baseline, get_stage3_sample_gray(sample), settings, cv2)
        focused_instability_mask = cv2.bitwise_and(instability_mask, art_state_footprint_mask)
        instability_scores.append(
            clamp_score(compute_binary_mask_ratio(focused_instability_mask) / max(settings.active_pixel_ratio * 2.0, 1e-6))
        )

    mean_persistence = sum(persistence_scores) / len(persistence_scores)
    mean_instability = sum(instability_scores) / len(instability_scores)
    return clamp_score(mean_persistence - (mean_instability * 0.5))



def compute_stage3_lasting_change_evidence_score(
    within_union_records: Iterable[MovementEvidenceRecord],
    candidate_union: CandidateUnion,
    after_reference_activity: float,
) -> float:
    persistent_difference_score = compute_persistent_difference_score(within_union_records)
    footprint_support_score = compute_footprint_support_score(
        records=within_union_records,
        footprint_size=candidate_union.union_footprint_size,
        footprint_scale=0.10,
    )
    after_window_persistence_score = compute_after_window_persistence_score(after_reference_activity)
    return clamp_score(
        (0.50 * persistent_difference_score)
        + (0.30 * after_window_persistence_score)
        + (0.20 * footprint_support_score)
    )



def compute_stage3_art_state_evidence_score(
    persistent_difference_score: float,
    footprint_support_score: float,
    after_window_persistence_score: float,
    reveal_window_hold_score: float,
) -> float:
    return clamp_score(
        (0.45 * persistent_difference_score)
        + (0.30 * footprint_support_score)
        + (0.15 * after_window_persistence_score)
        + (0.10 * reveal_window_hold_score)
    )



def compute_slice_lasting_change_evidence_score(
    slice_records: Iterable[MovementEvidenceRecord],
    footprint_size: int,
    after_reference_activity: float,
) -> float:
    persistent_difference_score = compute_persistent_difference_score(slice_records)
    footprint_support_score = compute_footprint_support_score(
        records=slice_records,
        footprint_size=footprint_size,
        footprint_scale=0.08,
    )
    after_window_persistence_score = compute_after_window_persistence_score(after_reference_activity)
    return clamp_score(
        (0.45 * persistent_difference_score)
        + (0.35 * footprint_support_score)
        + (0.20 * after_window_persistence_score)
    )



def screen_candidate_union(
    candidate_union: CandidateUnion,
    records: Iterable[MovementEvidenceRecord],
) -> ScreenedCandidateUnion:
    ordered_records = sorted(records, key=lambda record: record.frame_index)
    within_union_records = collect_records_in_frame_range(
        ordered_records,
        candidate_union.start_frame,
        candidate_union.end_frame + 1,
    )
    before_records = collect_records_in_frame_range(
        ordered_records,
        max(0, candidate_union.start_frame - STAGE3_REFERENCE_WINDOW_FRAMES),
        candidate_union.start_frame,
    )
    after_records = collect_records_in_frame_range(
        ordered_records,
        candidate_union.end_frame + 1,
        candidate_union.end_frame + 1 + STAGE3_REFERENCE_WINDOW_FRAMES,
    )

    mean_movement_strength = average_record_score(within_union_records, 'movement_strength_score')
    mean_temporal_persistence = average_record_score(within_union_records, 'temporal_persistence_score')
    mean_spatial_extent = average_record_score(within_union_records, 'spatial_extent_score')
    before_reference_activity = average_record_score(before_records, 'movement_strength_score')
    after_reference_activity = average_record_score(after_records, 'movement_strength_score')
    lasting_change_evidence_score = compute_stage3_lasting_change_evidence_score(
        within_union_records=within_union_records,
        candidate_union=candidate_union,
        after_reference_activity=after_reference_activity,
    )
    reference_windows_reliable = (
        len(before_records) >= STAGE3_MIN_REFERENCE_RECORDS
        and len(after_records) >= STAGE3_MIN_REFERENCE_RECORDS
    )
    reference_activity_ceiling = max(before_reference_activity, after_reference_activity)
    contrast_score = lasting_change_evidence_score - reference_activity_ceiling
    has_meaningful_union_activity = (
        bool(within_union_records)
        and candidate_union.union_footprint_size > 0
        and lasting_change_evidence_score >= STAGE3_SURVIVING_THRESHOLD
    )

    if not has_meaningful_union_activity:
        screening_result = 'rejected'
        surviving = False
        provisional_survival = False
        reason = 'weak_union_activity'
    elif not reference_windows_reliable:
        screening_result = 'provisional_surviving'
        surviving = True
        provisional_survival = True
        reason = 'reference_windows_unreliable'
    elif (
        reference_activity_ceiling <= STAGE3_MAX_REFERENCE_ACTIVITY
        or contrast_score >= STAGE3_MIN_CONTRAST_SCORE
    ):
        screening_result = 'surviving'
        surviving = True
        provisional_survival = False
        reason = 'union_activity_supported'
    else:
        screening_result = 'rejected'
        surviving = False
        provisional_survival = False
        reason = 'reference_windows_too_active'

    return ScreenedCandidateUnion(
        candidate_union=candidate_union,
        screening_result=screening_result,
        surviving=surviving,
        provisional_survival=provisional_survival,
        reason=reason,
        within_union_record_count=len(within_union_records),
        before_record_count=len(before_records),
        after_record_count=len(after_records),
        mean_movement_strength=mean_movement_strength,
        mean_temporal_persistence=mean_temporal_persistence,
        mean_spatial_extent=mean_spatial_extent,
        lasting_change_evidence_score=lasting_change_evidence_score,
        before_reference_activity=before_reference_activity,
        after_reference_activity=after_reference_activity,
        reference_windows_reliable=reference_windows_reliable,
    )



def build_stage3_cell_art_state_masks(
    footprint: frozenset[GridCoordinate],
    canvas_shape: tuple[int, int],
) -> dict[GridCoordinate, object]:
    return {
        coordinate: build_canvas_footprint_mask_from_coordinates([coordinate], canvas_shape)
        for coordinate in footprint
    }



def compute_stage3_cell_window_instability(
    window_samples: list[dict[str, object]],
    cell_art_state_mask,
    settings: DetectorSettings,
    cv2,
) -> float:
    if len(window_samples) < 2:
        return 0.0

    instability_scores: list[float] = []
    for sample_index in range(len(window_samples) - 1):
        instability_mask = build_art_state_change_mask(get_stage3_sample_gray(window_samples[sample_index]), get_stage3_sample_gray(window_samples[sample_index + 1]), settings, cv2)
        focused_instability_mask = cv2.bitwise_and(instability_mask, cell_art_state_mask)
        instability_scores.append(compute_binary_mask_ratio(focused_instability_mask))
    if not instability_scores:
        return 0.0
    return sum(instability_scores) / len(instability_scores)



def cell_has_meaningful_movement_in_window(
    ordered_records: list[MovementEvidenceRecord],
    window_start: int,
    window_end: int,
    coordinate: GridCoordinate,
    stage3_context: dict[str, object] | None = None,
) -> bool:
    if stage3_context is None:
        return any(
            record.movement_present and coordinate in record.touched_grid_coordinates
            for record in ordered_records
            if window_start <= record.frame_index < window_end
        )
    cache = stage3_context['cell_movement_cache']
    cache_key = (coordinate, window_start, window_end)
    if cache_key not in cache:
        cache[cache_key] = any(
            record.movement_present and coordinate in record.touched_grid_coordinates
            for record in get_stage3_window_records(stage3_context, ordered_records, window_start, window_end)
        )
    return cache[cache_key]



def window_has_meaningful_movement_outside_footprint(
    ordered_records: list[MovementEvidenceRecord],
    window_start: int,
    window_end: int,
    footprint: frozenset[GridCoordinate],
    stage3_context: dict[str, object] | None = None,
) -> bool:
    if stage3_context is None:
        return any(
            record.movement_present
            and any(coordinate not in footprint for coordinate in record.touched_grid_coordinates)
            for record in ordered_records
            if window_start <= record.frame_index < window_end
        )
    cache = stage3_context['outside_footprint_cache']
    cache_key = (footprint, window_start, window_end)
    if cache_key not in cache:
        cache[cache_key] = any(
            record.movement_present
            and any(coordinate not in footprint for coordinate in record.touched_grid_coordinates)
            for record in get_stage3_window_records(stage3_context, ordered_records, window_start, window_end)
        )
    return cache[cache_key]



def build_stage3_endpoint_coordinates(
    within_union_records: list[MovementEvidenceRecord],
) -> frozenset[GridCoordinate]:
    active_records = [
        record
        for record in within_union_records
        if record.movement_present and record.touched_grid_coordinates
    ]
    if not active_records:
        return frozenset()
    latest_frame = max(record.frame_index for record in active_records)
    return frozenset(
        coordinate
        for record in active_records
        if record.frame_index == latest_frame
        for coordinate in record.touched_grid_coordinates
    )



def build_stage3_cell_touch_frame_bounds(
    footprint: frozenset[GridCoordinate],
    within_union_records: list[MovementEvidenceRecord],
) -> dict[GridCoordinate, dict[str, int]]:
    bounds: dict[GridCoordinate, dict[str, int]] = {coordinate: {} for coordinate in footprint}
    for record in within_union_records:
        if not record.movement_present:
            continue
        for coordinate in record.touched_grid_coordinates:
            if coordinate not in bounds:
                continue
            cell_bounds = bounds[coordinate]
            cell_bounds.setdefault('first_touch_frame', record.frame_index)
            cell_bounds['last_touch_frame'] = record.frame_index
    return bounds



def window_has_meaningful_movement_outside_coordinate(
    ordered_records: list[MovementEvidenceRecord],
    window_start: int,
    window_end: int,
    coordinate: GridCoordinate,
    stage3_context: dict[str, object] | None = None,
) -> bool:
    if stage3_context is None:
        return any(
            record.movement_present
            and any(other_coordinate != coordinate for other_coordinate in record.touched_grid_coordinates)
            for record in ordered_records
            if window_start <= record.frame_index < window_end
        )
    cache = stage3_context['outside_coordinate_cache']
    cache_key = (coordinate, window_start, window_end)
    if cache_key not in cache:
        cache[cache_key] = any(
            record.movement_present
            and any(other_coordinate != coordinate for other_coordinate in record.touched_grid_coordinates)
            for record in get_stage3_window_records(stage3_context, ordered_records, window_start, window_end)
        )
    return cache[cache_key]



def stage3_cell_is_trustworthy_in_window(
    coordinate: GridCoordinate,
    window_start: int,
    window_end: int,
    ordered_records: list[MovementEvidenceRecord],
    sampled_frames: list[dict[str, object]],
    cell_art_state_masks: dict[GridCoordinate, object],
    footprint: frozenset[GridCoordinate],
    require_external_movement: bool,
    settings: DetectorSettings,
    cv2,
    stage3_context: dict[str, object] | None = None,
) -> bool:
    cache_key = None
    if stage3_context is not None:
        cache_key = (
            coordinate,
            window_start,
            window_end,
            require_external_movement,
            footprint if require_external_movement else None,
        )
        trust_cache = stage3_context['cell_trust_cache']
        if cache_key in trust_cache:
            return trust_cache[cache_key]

    window_samples = get_stage3_window_samples(stage3_context, sampled_frames, window_start, window_end)
    if len(window_samples) < STAGE3_ART_STATE_MIN_SAMPLES:
        result = False
    elif cell_has_meaningful_movement_in_window(ordered_records, window_start, window_end, coordinate, stage3_context):
        result = False
    elif require_external_movement and not window_has_meaningful_movement_outside_footprint(
        ordered_records,
        window_start,
        window_end,
        footprint,
        stage3_context,
    ):
        result = False
    else:
        instability = compute_stage3_cell_window_instability(
            window_samples,
            cell_art_state_masks[coordinate],
            settings,
            cv2,
        )
        result = instability <= STAGE3_ART_STATE_MAX_INTERNAL_INSTABILITY

    if stage3_context is not None and cache_key is not None:
        stage3_context['cell_trust_cache'][cache_key] = result
    return result

def build_stage3_reference_window_metadata(
    window_start: int,
    window_end: int,
    ordered_records: list[MovementEvidenceRecord],
    sampled_frames: list[dict[str, object]],
    stage3_context: dict[str, object] | None = None,
) -> dict[str, object]:
    if stage3_context is not None:
        cache = stage3_context['window_metadata_cache']
        cache_key = (window_start, window_end)
        if cache_key in cache:
            return dict(cache[cache_key])

    window_records = get_stage3_window_records(stage3_context, ordered_records, window_start, window_end)
    window_samples = get_stage3_window_samples(stage3_context, sampled_frames, window_start, window_end)
    metadata = {
        'window_start': window_start,
        'window_end': window_end,
        'mean_window_activity': average_record_score(window_records, 'movement_strength_score'),
        'sample_count': len(window_samples),
    }
    if stage3_context is not None:
        stage3_context['window_metadata_cache'][(window_start, window_end)] = dict(metadata)
    return metadata



def evaluate_stage3_full_footprint_window_blockers(
    *,
    window_start: int,
    window_end: int,
    footprint: frozenset[GridCoordinate],
    endpoint_coordinates: frozenset[GridCoordinate],
    require_external_movement_for_all_cells: bool,
    require_external_movement_for_endpoints: bool,
    ordered_records: list[MovementEvidenceRecord],
    sampled_frames: list[dict[str, object]],
    cell_art_state_masks: dict[GridCoordinate, object],
    settings: DetectorSettings,
    cv2,
    stage3_context: dict[str, object] | None = None,
) -> tuple[frozenset[GridCoordinate], frozenset[GridCoordinate]]:
    active_coordinates = get_stage3_window_active_coordinates(
        stage3_context,
        ordered_records,
        window_start,
        window_end,
    )
    blocking_coordinates: set[GridCoordinate] = set()
    active_blocking_coordinates: set[GridCoordinate] = set()
    for coordinate in footprint:
        require_external_movement = (
            require_external_movement_for_all_cells
            or (require_external_movement_for_endpoints and coordinate in endpoint_coordinates)
        )
        if stage3_cell_is_trustworthy_in_window(
            coordinate,
            window_start,
            window_end,
            ordered_records,
            sampled_frames,
            cell_art_state_masks,
            footprint,
            require_external_movement,
            settings,
            cv2,
            stage3_context,
        ):
            continue
        blocking_coordinates.add(coordinate)
        if coordinate in active_coordinates:
            active_blocking_coordinates.add(coordinate)
    return frozenset(blocking_coordinates), frozenset(active_blocking_coordinates)

def find_next_stage3_candidate_position_after_blocker_release(
    ordered_candidates: list[tuple[int, int]],
    current_candidate_position: int,
    active_blocking_coordinates: frozenset[GridCoordinate],
    ordered_records: list[MovementEvidenceRecord],
    stage3_context: dict[str, object] | None = None,
) -> int:
    if not active_blocking_coordinates:
        return current_candidate_position + 1

    next_candidate_position = current_candidate_position + 1
    while next_candidate_position < len(ordered_candidates):
        next_window_start, next_window_end = ordered_candidates[next_candidate_position]
        next_active_coordinates = get_stage3_window_active_coordinates(
            stage3_context,
            ordered_records,
            next_window_start,
            next_window_end,
        )
        if not active_blocking_coordinates.issubset(next_active_coordinates):
            return next_candidate_position
        next_candidate_position += 1
    return len(ordered_candidates)



def select_stage3_full_footprint_reference_window_v3(
    *,
    search_start: int,
    search_end: int,
    prefer_nearest: bool,
    footprint: frozenset[GridCoordinate],
    endpoint_coordinates: frozenset[GridCoordinate],
    require_external_movement_for_all_cells: bool,
    require_external_movement_for_endpoints: bool,
    require_external_movement_outside_coordinate_for_all_cells: bool,
    ordered_records: list[MovementEvidenceRecord],
    sampled_frames: list[dict[str, object]],
    cell_art_state_masks: dict[GridCoordinate, object],
    settings: DetectorSettings,
    cv2,
    stage3_context: dict[str, object] | None = None,
    status_callback: Callable[[str], None] | None = None,
    progress_prefix: str | None = None,
    progress_label: str = 'full-footprint window search progress',
) -> tuple[dict[str, object] | None, int]:
    candidate_windows = build_stage3_reference_window_candidates(
        search_start,
        search_end,
        STAGE3_ART_STATE_BEFORE_WINDOW_FRAMES,
        get_stage3_reference_candidate_step_frames(settings),
    )
    ordered_candidates = list(reversed(candidate_windows)) if prefer_nearest else candidate_windows
    total_candidates = len(ordered_candidates)
    progress_interval = max(1, total_candidates // 8) if total_candidates > 0 else 1
    progress_started_at = time.perf_counter() if status_callback is not None else 0.0
    candidate_position = 0
    checked_candidate_count = 0
    blocker_analysis_enabled = True

    while candidate_position < total_candidates:
        window_start, window_end = ordered_candidates[candidate_position]
        checked_candidate_count += 1

        blocking_coordinates: frozenset[GridCoordinate] = frozenset()
        active_blocking_coordinates: frozenset[GridCoordinate] = frozenset()
        if blocker_analysis_enabled:
            blocking_coordinates, active_blocking_coordinates = evaluate_stage3_full_footprint_window_blockers(
                window_start=window_start,
                window_end=window_end,
                footprint=footprint,
                endpoint_coordinates=endpoint_coordinates,
                require_external_movement_for_all_cells=require_external_movement_for_all_cells,
                require_external_movement_for_endpoints=require_external_movement_for_endpoints,
                ordered_records=ordered_records,
                sampled_frames=sampled_frames,
                cell_art_state_masks=cell_art_state_masks,
                settings=settings,
                cv2=cv2,
                stage3_context=stage3_context,
            )
            window_is_trustworthy = not blocking_coordinates
        else:
            window_is_trustworthy = all(
                stage3_cell_is_trustworthy_in_window(
                    coordinate,
                    window_start,
                    window_end,
                    ordered_records,
                    sampled_frames,
                    cell_art_state_masks,
                    footprint,
                    require_external_movement_for_all_cells
                    or (require_external_movement_for_endpoints and coordinate in endpoint_coordinates),
                    settings,
                    cv2,
                    stage3_context,
                )
                for coordinate in footprint
            )
        if window_is_trustworthy:
            if status_callback is not None and progress_prefix is not None:
                elapsed = format_stage_elapsed(time.perf_counter() - progress_started_at)
                status_callback(
                    f"{progress_prefix} - {progress_label}: "
                    f"{checked_candidate_count}/{max(1, total_candidates)} windows checked, "
                    f"full-footprint match found "
                    f"(elapsed {elapsed})"
                )
            metadata = build_stage3_reference_window_metadata(
                window_start,
                window_end,
                ordered_records,
                sampled_frames,
                stage3_context,
            )
            metadata['tier'] = 'full_footprint'
            return metadata, len(candidate_windows)

        next_candidate_position = candidate_position + 1
        jumped_window_count = 0
        if blocker_analysis_enabled and not active_blocking_coordinates:
            blocker_analysis_enabled = False
        elif blocker_analysis_enabled and active_blocking_coordinates and blocking_coordinates == active_blocking_coordinates:
            next_candidate_position = find_next_stage3_candidate_position_after_blocker_release(
                ordered_candidates,
                candidate_position,
                active_blocking_coordinates,
                ordered_records,
                stage3_context,
            )
            jumped_window_count = max(0, next_candidate_position - candidate_position - 1)

        if (
            status_callback is not None
            and progress_prefix is not None
            and (
                checked_candidate_count == 1
                or candidate_position == (total_candidates - 1)
                or (checked_candidate_count % progress_interval) == 0
                or jumped_window_count > 0
            )
        ):
            elapsed = format_stage_elapsed(time.perf_counter() - progress_started_at)
            jump_suffix = ''
            if jumped_window_count > 0:
                jump_suffix = f"; jumped over {jumped_window_count} windows until blocker activity changed"
            status_callback(
                f"{progress_prefix} - {progress_label}: "
                f"{checked_candidate_count}/{max(1, total_candidates)} windows checked, "
                f"no full-footprint match yet "
                f"({len(blocking_coordinates)} blocking cells, "
                f"{len(active_blocking_coordinates)} still movement-active; "
                f"elapsed {elapsed}{jump_suffix})"
            )
        candidate_position = next_candidate_position
    return None, len(candidate_windows)

def select_stage3_cell_reference_windows(
    *,
    search_start: int,
    search_end: int,
    prefer_nearest: bool,
    footprint: frozenset[GridCoordinate],
    endpoint_coordinates: frozenset[GridCoordinate],
    require_external_movement_for_all_cells: bool,
    require_external_movement_for_endpoints: bool,
    require_external_movement_outside_coordinate_for_all_cells: bool,
    ordered_records: list[MovementEvidenceRecord],
    sampled_frames: list[dict[str, object]],
    cell_art_state_masks: dict[GridCoordinate, object],
    settings: DetectorSettings,
    cv2,
    stage3_context: dict[str, object] | None = None,
    target_coordinates: frozenset[GridCoordinate] | None = None,
    status_callback: Callable[[str], None] | None = None,
    progress_prefix: str | None = None,
    progress_label: str = 'partial-footprint reference assignment',
) -> tuple[dict[GridCoordinate, dict[str, object]], int]:
    candidate_windows = build_stage3_reference_window_candidates(
        search_start,
        search_end,
        STAGE3_ART_STATE_BEFORE_WINDOW_FRAMES,
        get_stage3_reference_candidate_step_frames(settings),
    )
    ordered_candidates = list(reversed(candidate_windows)) if prefer_nearest else candidate_windows
    target_footprint = footprint if target_coordinates is None else target_coordinates
    assignments: dict[GridCoordinate, dict[str, object]] = {}
    total_candidates = len(ordered_candidates)
    total_target_cells = len(target_footprint)
    progress_interval = max(1, total_candidates // 8) if total_candidates > 0 else 1
    progress_started_at = time.perf_counter() if status_callback is not None else 0.0

    if require_external_movement_outside_coordinate_for_all_cells:
        search_active_coordinates = get_stage3_window_active_coordinates(
            stage3_context,
            ordered_records,
            search_start,
            search_end,
        )
        if not search_active_coordinates:
            if status_callback is not None and progress_prefix is not None:
                elapsed = format_stage_elapsed(time.perf_counter() - progress_started_at)
                status_callback(
                    f"{progress_prefix} - {progress_label}: "
                    f"0/{max(1, total_candidates)} windows checked, "
                    f"0/{max(1, total_target_cells)} cells assigned, "
                    f"{max(0, total_target_cells)} unresolved "
                    f"(elapsed {elapsed}; no movement evidence in search range, deferring to internal rescue)"
                )
            return assignments, len(candidate_windows)

    for candidate_index, (window_start, window_end) in enumerate(ordered_candidates, start=1):
        metadata: dict[str, object] | None = None
        active_coordinates = get_stage3_window_active_coordinates(
            stage3_context,
            ordered_records,
            window_start,
            window_end,
        )
        if require_external_movement_outside_coordinate_for_all_cells and not active_coordinates:
            if (
                status_callback is not None
                and progress_prefix is not None
                and (
                    candidate_index == 1
                    or candidate_index == total_candidates
                    or (candidate_index % progress_interval) == 0
                )
            ):
                elapsed = format_stage_elapsed(time.perf_counter() - progress_started_at)
                status_callback(
                    f"{progress_prefix} - {progress_label}: "
                    f"{candidate_index}/{max(1, total_candidates)} windows checked, "
                    f"{len(assignments)}/{max(1, total_target_cells)} cells assigned, "
                    f"{max(0, total_target_cells - len(assignments))} unresolved "
                    f"(elapsed {elapsed})"
                )
            continue
        for coordinate in target_footprint:
            if coordinate in assignments:
                continue
            if coordinate in active_coordinates:
                continue
            if stage3_cell_is_trustworthy_in_window(
                coordinate,
                window_start,
                window_end,
                ordered_records,
                sampled_frames,
                cell_art_state_masks,
                footprint,
                require_external_movement_for_all_cells or (require_external_movement_for_endpoints and coordinate in endpoint_coordinates),
                settings,
                cv2,
                stage3_context,
            ):
                if (
                    require_external_movement_outside_coordinate_for_all_cells
                    and not active_coordinates
                    and not window_has_meaningful_movement_outside_coordinate(
                        ordered_records,
                        window_start,
                        window_end,
                        coordinate,
                        stage3_context,
                    )
                ):
                    continue
                if metadata is None:
                    metadata = build_stage3_reference_window_metadata(
                        window_start,
                        window_end,
                        ordered_records,
                        sampled_frames,
                        stage3_context,
                    )
                    metadata['tier'] = 'cell_specific'
                assignments[coordinate] = metadata
        if (
            status_callback is not None
            and progress_prefix is not None
            and (
                candidate_index == 1
                or candidate_index == total_candidates
                or (candidate_index % progress_interval) == 0
            )
        ):
            elapsed = format_stage_elapsed(time.perf_counter() - progress_started_at)
            status_callback(
                f"{progress_prefix} - {progress_label}: "
                f"{candidate_index}/{max(1, total_candidates)} windows checked, "
                f"{len(assignments)}/{max(1, total_target_cells)} cells assigned, "
                f"{max(0, total_target_cells - len(assignments))} unresolved "
                f"(elapsed {elapsed})"
            )
        if len(assignments) == len(target_footprint):
            break
    return assignments, len(candidate_windows)



def select_stage3_internal_before_reference_windows(
    *,
    union_start: int,
    union_end: int,
    footprint: frozenset[GridCoordinate],
    touch_frame_bounds: dict[GridCoordinate, dict[str, int]],
    ordered_records: list[MovementEvidenceRecord],
    sampled_frames: list[dict[str, object]],
    cell_art_state_masks: dict[GridCoordinate, object],
    settings: DetectorSettings,
    cv2,
    stage3_context: dict[str, object] | None = None,
    target_coordinates: frozenset[GridCoordinate] | None = None,
) -> dict[GridCoordinate, dict[str, object]]:
    assignments: dict[GridCoordinate, dict[str, object]] = {}
    target_footprint = footprint if target_coordinates is None else target_coordinates
    for coordinate in target_footprint:
        first_touch_frame = touch_frame_bounds.get(coordinate, {}).get('first_touch_frame', union_end + 1)
        candidate_windows = build_stage3_reference_window_candidates(
            union_start,
            first_touch_frame,
            STAGE3_ART_STATE_BEFORE_WINDOW_FRAMES,
            get_stage3_reference_candidate_step_frames(settings),
        )
        for window_start, window_end in reversed(candidate_windows):
            if not stage3_cell_is_trustworthy_in_window(
                coordinate,
                window_start,
                window_end,
                ordered_records,
                sampled_frames,
                cell_art_state_masks,
                frozenset({coordinate}),
                False,
                settings,
                cv2,
                stage3_context,
            ):
                continue
            if not window_has_meaningful_movement_outside_coordinate(
                ordered_records,
                window_start,
                window_end,
                coordinate,
                stage3_context,
            ):
                continue
            metadata = build_stage3_reference_window_metadata(
                window_start,
                window_end,
                ordered_records,
                sampled_frames,
                stage3_context,
            )
            metadata['tier'] = 'internal_before'
            assignments[coordinate] = metadata
            break
    return assignments



def select_stage3_internal_after_reference_windows(
    *,
    union_start: int,
    union_end: int,
    search_end: int | None = None,
    footprint: frozenset[GridCoordinate],
    touch_frame_bounds: dict[GridCoordinate, dict[str, int]],
    ordered_records: list[MovementEvidenceRecord],
    sampled_frames: list[dict[str, object]],
    cell_art_state_masks: dict[GridCoordinate, object],
    settings: DetectorSettings,
    cv2,
    stage3_context: dict[str, object] | None = None,
    target_coordinates: frozenset[GridCoordinate] | None = None,
) -> dict[GridCoordinate, dict[str, object]]:
    assignments: dict[GridCoordinate, dict[str, object]] = {}
    target_footprint = footprint if target_coordinates is None else target_coordinates
    effective_search_end = union_end + 1 if search_end is None else max(union_end + 1, search_end)
    step_frames = max(1, get_stage3_reference_candidate_step_frames(settings))
    grouped_coordinates: dict[tuple[str, int], list[tuple[int, GridCoordinate]]] = {}

    for coordinate in sorted(target_footprint):
        cell_bounds = touch_frame_bounds.get(coordinate, {})
        if 'last_touch_frame' not in cell_bounds:
            continue
        search_start = cell_bounds['last_touch_frame'] + 1
        if (effective_search_end - search_start) <= STAGE3_ART_STATE_AFTER_WINDOW_FRAMES:
            group_key = ('exact', search_start)
        else:
            group_key = ('mod', search_start % step_frames)
        grouped_coordinates.setdefault(group_key, []).append((search_start, coordinate))

    ordered_groups = sorted(
        grouped_coordinates.values(),
        key=lambda coordinate_group: min(search_start for search_start, _ in coordinate_group),
    )
    for coordinate_group in ordered_groups:
        ordered_group_coordinates = sorted(coordinate_group, key=lambda item: (item[0], item[1]))
        group_search_start = ordered_group_coordinates[0][0]
        candidate_windows = build_stage3_reference_window_candidates(
            group_search_start,
            effective_search_end,
            STAGE3_ART_STATE_AFTER_WINDOW_FRAMES,
            step_frames,
        )
        unresolved_coordinates = {
            coordinate: search_start
            for search_start, coordinate in ordered_group_coordinates
        }
        for window_start, window_end in candidate_windows:
            eligible_coordinates = [
                coordinate
                for coordinate, coordinate_search_start in unresolved_coordinates.items()
                if coordinate_search_start <= window_start
            ]
            if not eligible_coordinates:
                continue
            metadata: dict[str, object] | None = None
            for coordinate in eligible_coordinates:
                if not stage3_cell_is_trustworthy_in_window(
                    coordinate,
                    window_start,
                    window_end,
                    ordered_records,
                    sampled_frames,
                    cell_art_state_masks,
                    frozenset({coordinate}),
                    False,
                    settings,
                    cv2,
                    stage3_context,
                ):
                    continue
                if not window_has_meaningful_movement_outside_coordinate(
                    ordered_records,
                    window_start,
                    window_end,
                    coordinate,
                    stage3_context,
                ):
                    continue
                if metadata is None:
                    metadata = build_stage3_reference_window_metadata(
                        window_start,
                        window_end,
                        ordered_records,
                        sampled_frames,
                        stage3_context,
                    )
                    metadata['tier'] = 'internal_after'
                assignments[coordinate] = metadata
            if assignments:
                unresolved_coordinates = {
                    coordinate: coordinate_search_start
                    for coordinate, coordinate_search_start in unresolved_coordinates.items()
                    if coordinate not in assignments
                }
            if not unresolved_coordinates:
                break
    return assignments


def select_stage3_composite_after_reference_windows(
    *,
    union_start: int,
    union_end: int,
    search_start: int,
    search_end: int,
    footprint: frozenset[GridCoordinate],
    endpoint_coordinates: frozenset[GridCoordinate],
    touch_frame_bounds: dict[GridCoordinate, dict[str, int]],
    ordered_records: list[MovementEvidenceRecord],
    sampled_frames: list[dict[str, object]],
    cell_art_state_masks: dict[GridCoordinate, object],
    settings: DetectorSettings,
    cv2,
    stage3_context: dict[str, object] | None = None,
    target_coordinates: frozenset[GridCoordinate] | None = None,
) -> tuple[dict[GridCoordinate, dict[str, object]], int, int]:
    target_footprint = footprint if target_coordinates is None else target_coordinates
    internal_assignments = select_stage3_internal_after_reference_windows(
        union_start=union_start,
        union_end=union_end,
        search_end=search_end,
        footprint=footprint,
        touch_frame_bounds=touch_frame_bounds,
        ordered_records=ordered_records,
        sampled_frames=sampled_frames,
        cell_art_state_masks=cell_art_state_masks,
        settings=settings,
        cv2=cv2,
        stage3_context=stage3_context,
        target_coordinates=target_footprint,
    )
    unresolved_coordinates = frozenset(target_footprint.difference(internal_assignments))
    external_assignments, external_candidate_count = ({}, 0) if not unresolved_coordinates else select_stage3_cell_reference_windows(
        search_start=search_start,
        search_end=search_end,
        prefer_nearest=False,
        footprint=footprint,
        endpoint_coordinates=endpoint_coordinates,
        require_external_movement_for_all_cells=False,
        require_external_movement_for_endpoints=True,
        require_external_movement_outside_coordinate_for_all_cells=False,
        ordered_records=ordered_records,
        sampled_frames=sampled_frames,
        cell_art_state_masks=cell_art_state_masks,
        settings=settings,
        cv2=cv2,
        stage3_context=stage3_context,
        target_coordinates=unresolved_coordinates,
    )
    return merge_stage3_reference_assignments(internal_assignments, external_assignments), len(internal_assignments), external_candidate_count


def merge_stage3_reference_assignments(
    primary_assignments: dict[GridCoordinate, dict[str, object]],
    secondary_assignments: dict[GridCoordinate, dict[str, object]],
) -> dict[GridCoordinate, dict[str, object]]:
    merged = dict(primary_assignments)
    for coordinate, metadata in secondary_assignments.items():
        merged.setdefault(coordinate, metadata)
    return merged


def get_stage3_window_baseline(
    baseline_cache: dict[tuple[int, int], tuple[list[dict[str, object]], object | None]],
    sampled_frames: list[dict[str, object]],
    window_start: int,
    window_end: int,
    stage3_context: dict[str, object] | None = None,
) -> tuple[list[dict[str, object]], object | None]:
    cache_key = (window_start, window_end)
    shared_baseline_cache = None if stage3_context is None else stage3_context.get('baseline_cache')
    if shared_baseline_cache is not None and cache_key in shared_baseline_cache:
        return shared_baseline_cache[cache_key]
    if cache_key not in baseline_cache:
        window_samples = get_stage3_window_samples(stage3_context, sampled_frames, window_start, window_end)
        baseline_cache[cache_key] = (window_samples, build_median_baseline(window_samples))
    if shared_baseline_cache is not None:
        shared_baseline_cache[cache_key] = baseline_cache[cache_key]
    return baseline_cache[cache_key]



def classify_stage3_cell_change(
    coordinate: GridCoordinate,
    before_window: dict[str, object],
    after_window: dict[str, object],
    sampled_frames: list[dict[str, object]],
    cell_art_state_masks: dict[GridCoordinate, object],
    baseline_cache: dict[tuple[int, int], tuple[list[dict[str, object]], object | None]],
    settings: DetectorSettings,
    cv2,
    stage3_context: dict[str, object] | None = None,
) -> str:
    _, before_baseline = get_stage3_window_baseline(
        baseline_cache,
        sampled_frames,
        int(before_window['window_start']),
        int(before_window['window_end']),
        stage3_context,
    )
    _, after_baseline = get_stage3_window_baseline(
        baseline_cache,
        sampled_frames,
        int(after_window['window_start']),
        int(after_window['window_end']),
        stage3_context,
    )
    if before_baseline is None or after_baseline is None:
        return 'resolved_ambiguous'

    change_mask = build_art_state_change_mask(before_baseline, after_baseline, settings, cv2)
    focused_change_mask = cv2.bitwise_and(change_mask, cell_art_state_masks[coordinate])
    change_score = compute_stage3_art_state_persistent_difference_score(
        focused_change_mask,
        1,
        settings,
    )
    return 'resolved_changed' if change_score >= 0.50 else 'resolved_unchanged'



def build_stage3_connected_clusters(
    coordinates: set[GridCoordinate],
) -> list[set[GridCoordinate]]:
    remaining = set(coordinates)
    clusters: list[set[GridCoordinate]] = []
    while remaining:
        start_coordinate = remaining.pop()
        cluster = {start_coordinate}
        pending = [start_coordinate]
        while pending:
            row_index, column_index = pending.pop()
            for row_offset, column_offset in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (row_index + row_offset, column_index + column_offset)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    cluster.add(neighbor)
                    pending.append(neighbor)
        clusters.append(cluster)
    return clusters



def evaluate_stage3_bucket_coverages(
    footprint: frozenset[GridCoordinate],
    bucket_by_cell: dict[GridCoordinate, str],
) -> dict[str, object]:
    total_cells = max(1, len(footprint))
    changed_cells = {coordinate for coordinate, bucket in bucket_by_cell.items() if bucket == 'resolved_changed'}
    unchanged_cells = {coordinate for coordinate, bucket in bucket_by_cell.items() if bucket == 'resolved_unchanged'}
    ambiguous_cells = {coordinate for coordinate, bucket in bucket_by_cell.items() if bucket == 'resolved_ambiguous'}
    changed_clusters = build_stage3_connected_clusters(changed_cells)
    ambiguous_clusters = build_stage3_connected_clusters(ambiguous_cells)
    return {
        'total_footprint_cells': total_cells,
        'resolved_changed_coverage': len(changed_cells) / total_cells,
        'resolved_unchanged_coverage': len(unchanged_cells) / total_cells,
        'resolved_ambiguous_coverage': len(ambiguous_cells) / total_cells,
        'largest_changed_cluster_size': max((len(cluster) for cluster in changed_clusters), default=0),
        'largest_changed_cluster_coverage': max((len(cluster) / total_cells for cluster in changed_clusters), default=0.0),
        'ambiguous_cluster_count': len(ambiguous_clusters),
    }



def decide_stage3_bucket_outcome(
    *,
    coverage_summary: dict[str, object],
    reconstructed_before_coverage: float,
    mode: str,
) -> tuple[str | None, bool, str | None, bool]:
    resolved_changed_coverage = float(coverage_summary['resolved_changed_coverage'])
    resolved_unchanged_coverage = float(coverage_summary['resolved_unchanged_coverage'])
    resolved_ambiguous_coverage = float(coverage_summary['resolved_ambiguous_coverage'])
    largest_changed_cluster_size = int(coverage_summary['largest_changed_cluster_size'])
    largest_changed_cluster_coverage = float(coverage_summary.get('largest_changed_cluster_coverage', 0.0))
    ambiguous_cluster_count = int(coverage_summary['ambiguous_cluster_count'])
    bounded_ambiguity = (
        resolved_ambiguous_coverage > 0.0
        and resolved_ambiguous_coverage <= 0.10
        and ambiguous_cluster_count == 1
    )
    rejection_priority = resolved_unchanged_coverage >= 0.85
    changed_evidence_survival = (
        reconstructed_before_coverage >= 0.80
        and resolved_changed_coverage >= 0.10
        and largest_changed_cluster_size >= 2
        and not rejection_priority
    )
    local_changed_cluster_survival = (
        reconstructed_before_coverage >= 0.80
        and largest_changed_cluster_size >= STAGE3_LOCAL_CHANGED_CLUSTER_MIN_SIZE
        and largest_changed_cluster_coverage >= STAGE3_LOCAL_CHANGED_CLUSTER_MIN_COVERAGE
        and resolved_ambiguous_coverage <= STAGE3_LOCAL_CHANGED_CLUSTER_MAX_AMBIGUOUS_COVERAGE
        and not rejection_priority
    )

    if changed_evidence_survival:
        if mode == 'step1':
            return 'surviving', True, 'step1_clear_survival', False
        return 'surviving', True, 'survived_by_changed_evidence', False
    if local_changed_cluster_survival:
        if mode == 'step1':
            return 'surviving', True, 'step1_local_changed_cluster_survival', False
        return 'surviving', True, 'survived_by_local_changed_cluster', False
    if bounded_ambiguity:
        return 'surviving', True, 'survived_by_bounded_ambiguity', True
    if rejection_priority:
        if mode == 'step1':
            return 'rejected', False, 'step1_clear_rejection', False
        return 'rejected', False, 'rejected_by_resolved_unchanged_evidence', False
    if mode == 'step1':
        return None, False, None, False
    return 'rejected', False, 'rejected_after_rescue_failure', False



def serialize_stage3_window_metadata(window_metadata: dict[str, object] | None) -> dict[str, object] | None:
    if window_metadata is None:
        return None
    return {
        'window_start': int(window_metadata['window_start']),
        'window_end': int(window_metadata['window_end']),
        'window_start_time': Timecode(total_frames=int(window_metadata['window_start'])).to_hhmmssff(),
        'window_end_time': Timecode(total_frames=int(window_metadata['window_end'])).to_hhmmssff(),
        'sample_count': int(window_metadata.get('sample_count', 0)),
        'mean_window_activity': round(float(window_metadata.get('mean_window_activity', 0.0)), 6),
        'tier': window_metadata.get('tier'),
    }



def serialize_stage3_coverage_summary(coverage_summary: dict[str, object]) -> dict[str, object]:
    return {
        'total_footprint_cells': int(coverage_summary.get('total_footprint_cells', 0)),
        'resolved_changed_coverage': round(float(coverage_summary.get('resolved_changed_coverage', 0.0)), 6),
        'resolved_unchanged_coverage': round(float(coverage_summary.get('resolved_unchanged_coverage', 0.0)), 6),
        'resolved_ambiguous_coverage': round(float(coverage_summary.get('resolved_ambiguous_coverage', 0.0)), 6),
        'largest_changed_cluster_size': int(coverage_summary.get('largest_changed_cluster_size', 0)),
        'largest_changed_cluster_coverage': round(float(coverage_summary.get('largest_changed_cluster_coverage', 0.0)), 6),
        'ambiguous_cluster_count': int(coverage_summary.get('ambiguous_cluster_count', 0)),
    }



def build_stage3_bucket_cell_rows(
    footprint: frozenset[GridCoordinate],
    bucket_by_cell: dict[GridCoordinate, str],
) -> list[dict[str, object]]:
    return [
        {
            'coordinate': [coordinate[0], coordinate[1]],
            'coordinate_label': format_grid_coordinate_label(coordinate),
            'bucket': bucket_by_cell.get(coordinate, 'resolved_ambiguous'),
        }
        for coordinate in sorted(footprint)
    ]



def build_stage3_rescue_cell_rows(
    footprint: frozenset[GridCoordinate],
    before_assignments: dict[GridCoordinate, dict[str, object]],
    after_assignments: dict[GridCoordinate, dict[str, object]],
    rescue_bucket_by_cell: dict[GridCoordinate, str],
) -> list[dict[str, object]]:
    return [
        {
            'coordinate': [coordinate[0], coordinate[1]],
            'coordinate_label': format_grid_coordinate_label(coordinate),
            'before_window': serialize_stage3_window_metadata(before_assignments.get(coordinate)),
            'after_window': serialize_stage3_window_metadata(after_assignments.get(coordinate)),
            'bucket': rescue_bucket_by_cell.get(coordinate, 'resolved_ambiguous'),
        }
        for coordinate in sorted(footprint)
    ]



def build_stage3_candidate_window_rows(candidate_windows: list[tuple[int, int]]) -> list[dict[str, object]]:
    return [
        {
            'window_start': int(window_start),
            'window_end': int(window_end),
            'window_start_time': Timecode(total_frames=int(window_start)).to_hhmmssff(),
            'window_end_time': Timecode(total_frames=int(window_end)).to_hhmmssff(),
        }
        for window_start, window_end in candidate_windows
    ]



def build_stage3_screened_union_result(
    *,
    candidate_union: CandidateUnion,
    within_union_records: list[MovementEvidenceRecord],
    before_records: list[MovementEvidenceRecord],
    after_records: list[MovementEvidenceRecord],
    mean_movement_strength: float,
    mean_temporal_persistence: float,
    mean_spatial_extent: float,
    before_reference_activity: float,
    after_reference_activity: float,
    screening_result: str,
    surviving: bool,
    reason: str,
    reference_windows_reliable: bool,
    stage3_mode: str,
    stage3_alignment_mode: str,
    resolved_changed_coverage: float,
    reconstructed_before_coverage: float,
    resolved_ambiguous_coverage: float,
    before_window: dict[str, object] | None,
    after_window: dict[str, object] | None,
    before_window_candidate_count: int,
    after_window_candidate_count: int,
    survived_by_bounded_ambiguity: bool,
    stage3_debug_trace: dict[str, object] | None = None,
) -> ScreenedCandidateUnion:
    return ScreenedCandidateUnion(
        candidate_union=candidate_union,
        screening_result=screening_result,
        surviving=surviving,
        provisional_survival=False,
        reason=reason,
        within_union_record_count=len(within_union_records),
        before_record_count=len(before_records),
        after_record_count=len(after_records),
        mean_movement_strength=mean_movement_strength,
        mean_temporal_persistence=mean_temporal_persistence,
        mean_spatial_extent=mean_spatial_extent,
        lasting_change_evidence_score=resolved_changed_coverage,
        before_reference_activity=before_reference_activity,
        after_reference_activity=after_reference_activity,
        reference_windows_reliable=reference_windows_reliable,
        stage3_mode=stage3_mode,
        stage3_alignment_mode=stage3_alignment_mode,
        stage3_persistent_difference_score=resolved_changed_coverage,
        stage3_footprint_support_score=reconstructed_before_coverage,
        stage3_after_window_persistence_score=1.0 - resolved_ambiguous_coverage,
        stage3_before_window_start=None if before_window is None else Timecode(total_frames=int(before_window['window_start'])).to_hhmmssff(),
        stage3_before_window_end=None if before_window is None else Timecode(total_frames=int(before_window['window_end'])).to_hhmmssff(),
        stage3_after_window_start=None if after_window is None else Timecode(total_frames=int(after_window['window_start'])).to_hhmmssff(),
        stage3_after_window_end=None if after_window is None else Timecode(total_frames=int(after_window['window_end'])).to_hhmmssff(),
        stage3_before_sample_count=0 if before_window is None else int(before_window['sample_count']),
        stage3_after_sample_count=0 if after_window is None else int(after_window['sample_count']),
        stage3_reveal_sample_count=0,
        stage3_before_window_quality_score=0.0 if before_window is None else 1.0,
        stage3_after_window_quality_score=0.0 if after_window is None else 1.0,
        stage3_reveal_window_quality_score=0.0,
        stage3_before_window_candidate_count=before_window_candidate_count,
        stage3_after_window_candidate_count=after_window_candidate_count,
        stage3_reveal_window_candidate_count=0,
        stage3_before_window_tier=None if before_window is None else 'full_footprint',
        stage3_after_window_tier=None if after_window is None else 'full_footprint',
        stage3_reveal_window_tier='bounded_ambiguity' if survived_by_bounded_ambiguity else None,
        stage3_reveal_window_hold_score=1.0 if survived_by_bounded_ambiguity else 0.0,
        stage3_debug_trace=stage3_debug_trace,
    )



def screen_candidate_union_with_art_state_prototype(
    candidate_union: CandidateUnion,
    records: list[MovementEvidenceRecord],
    sampled_frames: list[dict[str, object]],
    chapter_range: ChapterRange,
    settings: DetectorSettings,
    previous_union_end: int | None,
    next_union_start: int | None,
    cv2,
    stage3_context: dict[str, object] | None = None,
    status_callback: Callable[[str], None] | None = None,
    union_progress_prefix: str | None = None,
) -> ScreenedCandidateUnion:
    union_screen_started_at = time.perf_counter()
    stage3_timings: dict[str, float] = {}

    def record_stage3_timing(label: str, started_at: float) -> None:
        stage3_timings[label] = round(time.perf_counter() - started_at, 6)

    def finalize_stage3_debug_trace() -> None:
        stage3_timings['total_union_screen_seconds'] = round(time.perf_counter() - union_screen_started_at, 6)
        stage3_debug_trace['timings'] = dict(stage3_timings)

    def report_union_substep(message: str) -> None:
        if status_callback is None or union_progress_prefix is None:
            return
        status_callback(f"{union_progress_prefix} - {message}")

    ordered_records = records
    if stage3_context is None:
        stage3_context = build_stage3_runtime_context(ordered_records, sampled_frames)
    within_union_records = get_stage3_window_records(stage3_context, ordered_records,
        candidate_union.start_frame,
        candidate_union.end_frame + 1,
    )
    mean_movement_strength = average_record_score(within_union_records, 'movement_strength_score')
    mean_temporal_persistence = average_record_score(within_union_records, 'temporal_persistence_score')
    mean_spatial_extent = average_record_score(within_union_records, 'spatial_extent_score')

    stage3_debug_trace: dict[str, object] = {
        'candidate_union_index': candidate_union.union_index,
        'union_start_time': candidate_union.start_time,
        'union_end_time': candidate_union.end_time,
        'union_footprint_size': candidate_union.union_footprint_size,
        'previous_union_end': None if previous_union_end is None else Timecode(total_frames=previous_union_end).to_hhmmssff(),
        'next_union_start': None if next_union_start is None else Timecode(total_frames=next_union_start).to_hhmmssff(),
    }

    if not within_union_records or candidate_union.union_footprint_size <= 0 or not sampled_frames:
        stage3_debug_trace['step1'] = {'attempted': False, 'reason': 'missing_union_activity_or_samples'}
        stage3_debug_trace['snapshot_rescue'] = {'attempted': False, 'reason': 'missing_union_activity_or_samples'}
        stage3_debug_trace['final_stage3_outcome'] = {
            'screening_result': 'rejected',
            'surviving': False,
            'reason': 'weak_union_activity',
            'mode': 'snapshot_rescue',
            'alignment_mode': 'none',
        }
        finalize_stage3_debug_trace()
        return build_stage3_screened_union_result(
            candidate_union=candidate_union,
            within_union_records=within_union_records,
            before_records=[],
            after_records=[],
            mean_movement_strength=mean_movement_strength,
            mean_temporal_persistence=mean_temporal_persistence,
            mean_spatial_extent=mean_spatial_extent,
            before_reference_activity=0.0,
            after_reference_activity=0.0,
            screening_result='rejected',
            surviving=False,
            reason='weak_union_activity',
            reference_windows_reliable=False,
            stage3_mode='snapshot_rescue',
            stage3_alignment_mode='none',
            resolved_changed_coverage=0.0,
            reconstructed_before_coverage=0.0,
            resolved_ambiguous_coverage=1.0 if candidate_union.union_footprint_size > 0 else 0.0,
            before_window=None,
            after_window=None,
            before_window_candidate_count=0,
            after_window_candidate_count=0,
            survived_by_bounded_ambiguity=False,
            stage3_debug_trace=stage3_debug_trace,
        )

    setup_started_at = time.perf_counter()
    reference_canvas_shape = get_stage3_canvas_shape(sampled_frames[0])
    cell_art_state_masks = build_stage3_cell_art_state_masks(candidate_union.union_footprint, reference_canvas_shape)
    endpoint_coordinates = build_stage3_endpoint_coordinates(within_union_records)
    touch_frame_bounds = build_stage3_cell_touch_frame_bounds(candidate_union.union_footprint, within_union_records)
    baseline_cache: dict[tuple[int, int], tuple[list[dict[str, object]], object | None]] = {}

    simple_before_search_start = max(chapter_range.start.total_frames, candidate_union.start_frame - STAGE3_ART_STATE_SIMPLE_SEARCH_FRAMES)
    simple_before_search_end = max(simple_before_search_start, candidate_union.start_frame - STAGE3_ART_STATE_BEFORE_OFFSET_FRAMES)
    simple_after_search_start = min(chapter_range.end.total_frames, candidate_union.end_frame + STAGE3_ART_STATE_AFTER_DELAY_FRAMES)
    simple_after_search_end = min(chapter_range.end.total_frames, candidate_union.end_frame + STAGE3_ART_STATE_SIMPLE_SEARCH_FRAMES)
    if next_union_start is not None:
        simple_after_search_end = min(simple_after_search_end, next_union_start)

    rescue_before_search_start = max(chapter_range.start.total_frames, candidate_union.start_frame - STAGE3_ART_STATE_RESCUE_SEARCH_FRAMES)
    rescue_before_search_end = simple_before_search_end
    rescue_after_search_start = simple_after_search_start
    rescue_after_search_end = min(chapter_range.end.total_frames, candidate_union.end_frame + STAGE3_ART_STATE_RESCUE_SEARCH_FRAMES)
    if next_union_start is not None:
        rescue_after_search_end = min(rescue_after_search_end, next_union_start)

    stage3_debug_trace['search_ranges'] = {
        'step1_before_search_start': Timecode(total_frames=simple_before_search_start).to_hhmmssff(),
        'step1_before_search_end': Timecode(total_frames=simple_before_search_end).to_hhmmssff(),
        'step1_after_search_start': Timecode(total_frames=simple_after_search_start).to_hhmmssff(),
        'step1_after_search_end': Timecode(total_frames=simple_after_search_end).to_hhmmssff(),
        'rescue_before_search_start': Timecode(total_frames=rescue_before_search_start).to_hhmmssff(),
        'rescue_before_search_end': Timecode(total_frames=rescue_before_search_end).to_hhmmssff(),
        'rescue_after_search_start': Timecode(total_frames=rescue_after_search_start).to_hhmmssff(),
        'rescue_after_search_end': Timecode(total_frames=rescue_after_search_end).to_hhmmssff(),
        'step1_before_candidate_windows': build_stage3_candidate_window_rows(
            build_stage3_reference_window_candidates(simple_before_search_start, simple_before_search_end, STAGE3_ART_STATE_BEFORE_WINDOW_FRAMES, get_stage3_reference_candidate_step_frames(settings))
        ),
        'step1_after_candidate_windows': build_stage3_candidate_window_rows(
            build_stage3_reference_window_candidates(simple_after_search_start, simple_after_search_end, STAGE3_ART_STATE_AFTER_WINDOW_FRAMES, get_stage3_reference_candidate_step_frames(settings))
        ),
        'rescue_before_candidate_windows': build_stage3_candidate_window_rows(
            build_stage3_reference_window_candidates(rescue_before_search_start, rescue_before_search_end, STAGE3_ART_STATE_BEFORE_WINDOW_FRAMES, get_stage3_reference_candidate_step_frames(settings))
        ),
        'rescue_after_candidate_windows': build_stage3_candidate_window_rows(
            build_stage3_reference_window_candidates(rescue_after_search_start, rescue_after_search_end, STAGE3_ART_STATE_AFTER_WINDOW_FRAMES, get_stage3_reference_candidate_step_frames(settings))
        ),
    }

    record_stage3_timing('setup_seconds', setup_started_at)
    step1_before_search_started_at = time.perf_counter()
    report_union_substep('Step 1 full-footprint window search')
    selected_before_window, before_window_candidate_count = select_stage3_full_footprint_reference_window_v3(
        search_start=simple_before_search_start,
        search_end=simple_before_search_end,
        prefer_nearest=True,
        footprint=candidate_union.union_footprint,
        endpoint_coordinates=endpoint_coordinates,
        require_external_movement_for_all_cells=False,
        require_external_movement_for_endpoints=False,
        require_external_movement_outside_coordinate_for_all_cells=True,
        ordered_records=ordered_records,
        sampled_frames=sampled_frames,
        cell_art_state_masks=cell_art_state_masks,
        settings=settings,
        cv2=cv2,
        stage3_context=stage3_context,
        status_callback=status_callback,
        progress_prefix=union_progress_prefix,
        progress_label='Step 1 full-footprint before search progress',
    )
    record_stage3_timing('step1_full_before_search_seconds', step1_before_search_started_at)
    step1_after_fast_path_attempted = (
        candidate_union.union_footprint_size <= STAGE3_FULL_AFTER_FAST_PATH_MAX_FOOTPRINT
    )
    after_window_candidate_count = len(
        build_stage3_reference_window_candidates(
            simple_after_search_start,
            simple_after_search_end,
            STAGE3_ART_STATE_AFTER_WINDOW_FRAMES,
            get_stage3_reference_candidate_step_frames(settings),
        )
    )
    selected_after_window = None
    if step1_after_fast_path_attempted:
        step1_after_search_started_at = time.perf_counter()
        selected_after_window, after_window_candidate_count = select_stage3_full_footprint_reference_window_v3(
            search_start=simple_after_search_start,
            search_end=simple_after_search_end,
            prefer_nearest=False,
            footprint=candidate_union.union_footprint,
            endpoint_coordinates=endpoint_coordinates,
            require_external_movement_for_all_cells=False,
            require_external_movement_for_endpoints=True,
            require_external_movement_outside_coordinate_for_all_cells=False,
            ordered_records=ordered_records,
            sampled_frames=sampled_frames,
            cell_art_state_masks=cell_art_state_masks,
            settings=settings,
            cv2=cv2,
            stage3_context=stage3_context,
            status_callback=status_callback,
            progress_prefix=union_progress_prefix,
            progress_label='Step 1 full-footprint after search progress',
        )
        record_stage3_timing('step1_full_after_search_seconds', step1_after_search_started_at)
    else:
        stage3_timings['step1_full_after_search_seconds'] = 0.0

    report_union_substep(
        f"Step 1 full-footprint search complete "
        f"(before candidates {before_window_candidate_count}, after candidates {after_window_candidate_count})"
    )

    step1_trace: dict[str, object] = {
        'attempted': True,
        'full_footprint': {
            'selected_before_window': serialize_stage3_window_metadata(selected_before_window),
            'selected_after_window': serialize_stage3_window_metadata(selected_after_window),
            'before_window_candidate_count': before_window_candidate_count,
            'after_window_candidate_count': after_window_candidate_count,
            'after_fast_path_attempted': step1_after_fast_path_attempted,
        },
    }

    if selected_before_window is not None and selected_after_window is not None:
        report_union_substep('Step 1 full-footprint cell classification')
        step1_full_classification_started_at = time.perf_counter()
        step1_bucket_by_cell = {
            coordinate: classify_stage3_cell_change(
                coordinate,
                selected_before_window,
                selected_after_window,
                sampled_frames,
                cell_art_state_masks,
                baseline_cache,
                settings,
                cv2,
                stage3_context,
            )
            for coordinate in candidate_union.union_footprint
        }
        record_stage3_timing('step1_full_classification_seconds', step1_full_classification_started_at)
        step1_coverage = evaluate_stage3_bucket_coverages(candidate_union.union_footprint, step1_bucket_by_cell)
        step1_screening_result, step1_surviving, step1_reason, step1_bounded_ambiguity = decide_stage3_bucket_outcome(
            coverage_summary=step1_coverage,
            reconstructed_before_coverage=1.0,
            mode='step1',
        )
        step1_trace['full_footprint']['bucket_coverages'] = serialize_stage3_coverage_summary(step1_coverage)
        step1_trace['full_footprint']['bucket_by_cell'] = build_stage3_bucket_cell_rows(candidate_union.union_footprint, step1_bucket_by_cell)
        step1_trace['full_footprint']['decision'] = {
            'screening_result': step1_screening_result,
            'surviving': step1_surviving,
            'reason': step1_reason,
            'bounded_ambiguity': step1_bounded_ambiguity,
        }
        if step1_screening_result is not None and step1_reason is not None:
            stage3_debug_trace['step1'] = step1_trace
            stage3_debug_trace['snapshot_rescue'] = {'attempted': False, 'reason': 'step1_reached_decision'}
            stage3_debug_trace['final_stage3_outcome'] = {
                'screening_result': step1_screening_result,
                'surviving': step1_surviving,
                'reason': step1_reason,
                'mode': 'step1',
                'alignment_mode': 'full_footprint',
            }
            before_records = get_stage3_window_records(stage3_context, ordered_records,
                int(selected_before_window['window_start']),
                int(selected_before_window['window_end']),
            )
            after_records = get_stage3_window_records(stage3_context, ordered_records,
                int(selected_after_window['window_start']),
                int(selected_after_window['window_end']),
            )
            finalize_stage3_debug_trace()
            return build_stage3_screened_union_result(
                candidate_union=candidate_union,
                within_union_records=within_union_records,
                before_records=before_records,
                after_records=after_records,
                mean_movement_strength=mean_movement_strength,
                mean_temporal_persistence=mean_temporal_persistence,
                mean_spatial_extent=mean_spatial_extent,
                before_reference_activity=float(selected_before_window['mean_window_activity']),
                after_reference_activity=float(selected_after_window['mean_window_activity']),
                screening_result=step1_screening_result,
                surviving=step1_surviving,
                reason=step1_reason,
                reference_windows_reliable=True,
                stage3_mode='step1',
                stage3_alignment_mode='full_footprint',
                resolved_changed_coverage=float(step1_coverage['resolved_changed_coverage']),
                reconstructed_before_coverage=1.0,
                resolved_ambiguous_coverage=float(step1_coverage['resolved_ambiguous_coverage']),
                before_window=selected_before_window,
                after_window=selected_after_window,
                before_window_candidate_count=before_window_candidate_count,
                after_window_candidate_count=after_window_candidate_count,
                survived_by_bounded_ambiguity=step1_bounded_ambiguity,
                stage3_debug_trace=stage3_debug_trace,
            )
    else:
        step1_trace['full_footprint']['bucket_coverages'] = None
        step1_trace['full_footprint']['bucket_by_cell'] = []
        step1_trace['full_footprint']['decision'] = {
            'screening_result': None,
            'surviving': False,
            'reason': 'full_footprint_windows_not_found',
            'bounded_ambiguity': False,
        }

    report_union_substep('Step 1 partial-footprint reference assignment')
    step1_partial_before_assignment_started_at = time.perf_counter()
    step1_before_assignments: dict[GridCoordinate, dict[str, object]]
    step1_before_candidate_count = before_window_candidate_count
    if selected_before_window is not None and not step1_after_fast_path_attempted:
        step1_before_assignments = {
            coordinate: dict(selected_before_window)
            for coordinate in candidate_union.union_footprint
        }
        stage3_timings['step1_partial_before_assignment_seconds'] = 0.0
    else:
        step1_before_assignments, step1_before_candidate_count = select_stage3_cell_reference_windows(
            search_start=simple_before_search_start,
            search_end=simple_before_search_end,
            prefer_nearest=True,
            footprint=candidate_union.union_footprint,
            endpoint_coordinates=endpoint_coordinates,
            require_external_movement_for_all_cells=False,
            require_external_movement_for_endpoints=False,
            require_external_movement_outside_coordinate_for_all_cells=True,
            ordered_records=ordered_records,
            sampled_frames=sampled_frames,
            cell_art_state_masks=cell_art_state_masks,
            settings=settings,
            cv2=cv2,
            stage3_context=stage3_context,
            status_callback=status_callback,
            progress_prefix=union_progress_prefix,
            progress_label='Step 1 partial before assignment progress',
        )
        record_stage3_timing('step1_partial_before_assignment_seconds', step1_partial_before_assignment_started_at)
    step1_partial_after_assignment_started_at = time.perf_counter()
    if step1_after_fast_path_attempted:
        step1_internal_after_assignment_count = 0
        step1_after_assignments, step1_after_candidate_count = select_stage3_cell_reference_windows(
            search_start=simple_after_search_start,
            search_end=simple_after_search_end,
            prefer_nearest=False,
            footprint=candidate_union.union_footprint,
            endpoint_coordinates=endpoint_coordinates,
            require_external_movement_for_all_cells=False,
            require_external_movement_for_endpoints=True,
            require_external_movement_outside_coordinate_for_all_cells=False,
            ordered_records=ordered_records,
            sampled_frames=sampled_frames,
            cell_art_state_masks=cell_art_state_masks,
            settings=settings,
            cv2=cv2,
            stage3_context=stage3_context,
            status_callback=status_callback,
            progress_prefix=union_progress_prefix,
            progress_label='Step 1 partial after assignment progress',
        )
    else:
        step1_after_assignments, step1_internal_after_assignment_count, step1_after_candidate_count = select_stage3_composite_after_reference_windows(
            union_start=candidate_union.start_frame,
            union_end=candidate_union.end_frame,
            search_start=simple_after_search_start,
            search_end=simple_after_search_end,
            footprint=candidate_union.union_footprint,
            endpoint_coordinates=endpoint_coordinates,
            touch_frame_bounds=touch_frame_bounds,
            ordered_records=ordered_records,
            sampled_frames=sampled_frames,
            cell_art_state_masks=cell_art_state_masks,
            settings=settings,
            cv2=cv2,
            stage3_context=stage3_context,
        )
    record_stage3_timing('step1_partial_after_assignment_seconds', step1_partial_after_assignment_started_at)
    report_union_substep(
        f"Step 1 partial-footprint assignment complete "
        f"(before assignments {len(step1_before_assignments)}, after assignments {len(step1_after_assignments)})"
    )
    report_union_substep('Step 1 partial-footprint cell classification')
    step1_partial_classification_started_at = time.perf_counter()
    step1_partial_bucket_by_cell: dict[GridCoordinate, str] = {}
    for coordinate in candidate_union.union_footprint:
        before_window = step1_before_assignments.get(coordinate)
        after_window = step1_after_assignments.get(coordinate)
        if before_window is None or after_window is None:
            step1_partial_bucket_by_cell[coordinate] = 'resolved_ambiguous'
            continue
        step1_partial_bucket_by_cell[coordinate] = classify_stage3_cell_change(
            coordinate,
            before_window,
            after_window,
            sampled_frames,
            cell_art_state_masks,
            baseline_cache,
            settings,
            cv2,
            stage3_context,
        )
    record_stage3_timing('step1_partial_classification_seconds', step1_partial_classification_started_at)
    step1_partial_coverage = evaluate_stage3_bucket_coverages(candidate_union.union_footprint, step1_partial_bucket_by_cell)
    step1_partial_before_coverage = 1.0 if (selected_before_window is not None and not step1_after_fast_path_attempted) else (len(step1_before_assignments) / max(1, candidate_union.union_footprint_size))
    step1_partial_screening_result, step1_partial_surviving, step1_partial_reason, step1_partial_bounded_ambiguity = decide_stage3_bucket_outcome(
        coverage_summary=step1_partial_coverage,
        reconstructed_before_coverage=step1_partial_before_coverage,
        mode='step1',
    )
    step1_trace['partial_footprint'] = {
        'before_assignment_count': len(step1_before_assignments),
        'after_assignment_count': len(step1_after_assignments),
        'before_window_candidate_count': step1_before_candidate_count,
        'after_window_candidate_count': step1_after_candidate_count,
        'internal_after_assignment_count': step1_internal_after_assignment_count,
        'before_assignment_source': 'shared_full_footprint' if (selected_before_window is not None and not step1_after_fast_path_attempted) else 'cell_specific',
        'representative_before_window': serialize_stage3_window_metadata(next(iter(step1_before_assignments.values())) if step1_before_assignments else None),
        'representative_after_window': serialize_stage3_window_metadata(next(iter(step1_after_assignments.values())) if step1_after_assignments else None),
        'reconstructed_before_coverage': round(step1_partial_before_coverage, 6),
        'bucket_coverages': serialize_stage3_coverage_summary(step1_partial_coverage),
        'bucket_by_cell': build_stage3_bucket_cell_rows(candidate_union.union_footprint, step1_partial_bucket_by_cell),
        'decision': {
            'screening_result': step1_partial_screening_result,
            'surviving': step1_partial_surviving,
            'reason': step1_partial_reason,
            'bounded_ambiguity': step1_partial_bounded_ambiguity,
        },
    }
    if step1_partial_screening_result is not None and step1_partial_reason is not None:
        representative_before_window = next(iter(step1_before_assignments.values())) if step1_before_assignments else None
        representative_after_window = next(iter(step1_after_assignments.values())) if step1_after_assignments else None
        stage3_debug_trace['step1'] = step1_trace
        stage3_debug_trace['snapshot_rescue'] = {'attempted': False, 'reason': 'step1_partial_reached_decision'}
        stage3_debug_trace['final_stage3_outcome'] = {
            'screening_result': step1_partial_screening_result,
            'surviving': step1_partial_surviving,
            'reason': step1_partial_reason,
            'mode': 'step1',
            'alignment_mode': 'partial_footprint',
        }
        before_records = [] if representative_before_window is None else get_stage3_window_records(stage3_context, ordered_records,
            int(representative_before_window['window_start']),
            int(representative_before_window['window_end']),
        )
        after_records = [] if representative_after_window is None else get_stage3_window_records(stage3_context, ordered_records,
            int(representative_after_window['window_start']),
            int(representative_after_window['window_end']),
        )
        finalize_stage3_debug_trace()
        return build_stage3_screened_union_result(
            candidate_union=candidate_union,
            within_union_records=within_union_records,
            before_records=before_records,
            after_records=after_records,
            mean_movement_strength=mean_movement_strength,
            mean_temporal_persistence=mean_temporal_persistence,
            mean_spatial_extent=mean_spatial_extent,
            before_reference_activity=0.0 if representative_before_window is None else float(representative_before_window['mean_window_activity']),
            after_reference_activity=0.0 if representative_after_window is None else float(representative_after_window['mean_window_activity']),
            screening_result=step1_partial_screening_result,
            surviving=step1_partial_surviving,
            reason=step1_partial_reason,
            reference_windows_reliable=bool(step1_before_assignments) and bool(step1_after_assignments),
            stage3_mode='step1',
            stage3_alignment_mode='partial_footprint',
            resolved_changed_coverage=float(step1_partial_coverage['resolved_changed_coverage']),
            reconstructed_before_coverage=step1_partial_before_coverage,
            resolved_ambiguous_coverage=float(step1_partial_coverage['resolved_ambiguous_coverage']),
            before_window=representative_before_window,
            after_window=representative_after_window,
            before_window_candidate_count=step1_before_candidate_count,
            after_window_candidate_count=step1_after_candidate_count,
            survived_by_bounded_ambiguity=step1_partial_bounded_ambiguity,
            stage3_debug_trace=stage3_debug_trace,
        )

    stage3_debug_trace['step1'] = step1_trace
    before_assignments = dict(step1_before_assignments)
    after_assignments = dict(step1_after_assignments)
    unresolved_before_coordinates = frozenset(candidate_union.union_footprint.difference(before_assignments))
    report_union_substep('Snapshot rescue before-state reference assignment')
    rescue_before_internal_assignment_started_at = time.perf_counter()
    internal_before_assignments = {} if not unresolved_before_coordinates else select_stage3_internal_before_reference_windows(
        union_start=candidate_union.start_frame,
        union_end=candidate_union.end_frame,
        footprint=candidate_union.union_footprint,
        touch_frame_bounds=touch_frame_bounds,
        ordered_records=ordered_records,
        sampled_frames=sampled_frames,
        cell_art_state_masks=cell_art_state_masks,
        settings=settings,
        cv2=cv2,
        stage3_context=stage3_context,
        target_coordinates=unresolved_before_coordinates,
    )
    before_assignments = merge_stage3_reference_assignments(before_assignments, internal_before_assignments)
    record_stage3_timing('rescue_before_internal_assignment_seconds', rescue_before_internal_assignment_started_at)
    unresolved_before_coordinates = frozenset(candidate_union.union_footprint.difference(before_assignments))
    rescue_before_external_assignment_started_at = time.perf_counter()
    external_before_assignments, rescue_before_candidate_count = ({}, 0) if not unresolved_before_coordinates else select_stage3_cell_reference_windows(
        search_start=rescue_before_search_start,
        search_end=rescue_before_search_end,
        prefer_nearest=True,
        footprint=candidate_union.union_footprint,
        endpoint_coordinates=endpoint_coordinates,
        require_external_movement_for_all_cells=False,
        require_external_movement_for_endpoints=False,
        require_external_movement_outside_coordinate_for_all_cells=True,
        ordered_records=ordered_records,
        sampled_frames=sampled_frames,
        cell_art_state_masks=cell_art_state_masks,
        settings=settings,
        cv2=cv2,
        stage3_context=stage3_context,
        target_coordinates=unresolved_before_coordinates,
    )
    before_assignments = merge_stage3_reference_assignments(before_assignments, external_before_assignments)
    record_stage3_timing('rescue_before_external_assignment_seconds', rescue_before_external_assignment_started_at)

    report_union_substep(
        f"Snapshot rescue before-state assignment complete "
        f"(internal {len(internal_before_assignments)}, external {len(external_before_assignments)})"
    )

    after_assignments = dict(step1_after_assignments)
    unresolved_after_coordinates = frozenset(candidate_union.union_footprint.difference(after_assignments))
    report_union_substep('Snapshot rescue after-state reference assignment')
    rescue_after_internal_assignment_started_at = time.perf_counter()
    internal_after_assignments = {} if not unresolved_after_coordinates else select_stage3_internal_after_reference_windows(
        union_start=candidate_union.start_frame,
        union_end=candidate_union.end_frame,
        footprint=candidate_union.union_footprint,
        touch_frame_bounds=touch_frame_bounds,
        ordered_records=ordered_records,
        sampled_frames=sampled_frames,
        cell_art_state_masks=cell_art_state_masks,
        settings=settings,
        cv2=cv2,
        stage3_context=stage3_context,
        target_coordinates=unresolved_after_coordinates,
    )
    after_assignments = merge_stage3_reference_assignments(after_assignments, internal_after_assignments)
    record_stage3_timing('rescue_after_internal_assignment_seconds', rescue_after_internal_assignment_started_at)
    unresolved_after_coordinates = frozenset(candidate_union.union_footprint.difference(after_assignments))
    rescue_after_external_assignment_started_at = time.perf_counter()
    external_after_assignments, rescue_after_candidate_count = ({}, 0) if not unresolved_after_coordinates else select_stage3_cell_reference_windows(
        search_start=rescue_after_search_start,
        search_end=rescue_after_search_end,
        prefer_nearest=False,
        footprint=candidate_union.union_footprint,
        endpoint_coordinates=endpoint_coordinates,
        require_external_movement_for_all_cells=False,
        require_external_movement_for_endpoints=True,
        require_external_movement_outside_coordinate_for_all_cells=False,
        ordered_records=ordered_records,
        sampled_frames=sampled_frames,
        cell_art_state_masks=cell_art_state_masks,
        settings=settings,
        cv2=cv2,
        stage3_context=stage3_context,
        target_coordinates=unresolved_after_coordinates,
    )
    after_assignments = merge_stage3_reference_assignments(after_assignments, external_after_assignments)
    record_stage3_timing('rescue_after_external_assignment_seconds', rescue_after_external_assignment_started_at)

    report_union_substep(
        f"Snapshot rescue after-state assignment complete "
        f"(internal {len(internal_after_assignments)}, external {len(external_after_assignments)})"
    )
    report_union_substep('Snapshot rescue cell classification')
    rescue_classification_started_at = time.perf_counter()
    rescue_bucket_by_cell: dict[GridCoordinate, str] = {}
    for coordinate in candidate_union.union_footprint:
        before_window = before_assignments.get(coordinate)
        after_window = after_assignments.get(coordinate)
        if before_window is None or after_window is None:
            rescue_bucket_by_cell[coordinate] = 'resolved_ambiguous'
            continue
        rescue_bucket_by_cell[coordinate] = classify_stage3_cell_change(
            coordinate,
            before_window,
            after_window,
            sampled_frames,
            cell_art_state_masks,
            baseline_cache,
            settings,
            cv2,
            stage3_context,
        )

    record_stage3_timing('rescue_classification_seconds', rescue_classification_started_at)
    rescue_coverage = evaluate_stage3_bucket_coverages(candidate_union.union_footprint, rescue_bucket_by_cell)
    reconstructed_before_coverage = len(before_assignments) / max(1, candidate_union.union_footprint_size)
    rescue_screening_result, rescue_surviving, rescue_reason, rescue_bounded_ambiguity = decide_stage3_bucket_outcome(
        coverage_summary=rescue_coverage,
        reconstructed_before_coverage=reconstructed_before_coverage,
        mode='snapshot_rescue',
    )
    if rescue_screening_result is None or rescue_reason is None:
        rescue_screening_result = 'rejected'
        rescue_surviving = False
        rescue_reason = 'rejected_after_rescue_failure'

    representative_before_window = selected_before_window
    if representative_before_window is None and before_assignments:
        representative_before_window = next(iter(before_assignments.values()))
    representative_after_window = selected_after_window
    if representative_after_window is None and after_assignments:
        representative_after_window = next(iter(after_assignments.values()))

    stage3_debug_trace['snapshot_rescue'] = {
        'attempted': True,
        'internal_before_assignment_count': len(internal_before_assignments),
        'external_before_assignment_count': len(external_before_assignments),
        'before_assignment_count': len(before_assignments),
        'internal_after_assignment_count': len(internal_after_assignments),
        'external_after_assignment_count': len(external_after_assignments),
        'after_assignment_count': len(after_assignments),
        'before_window_candidate_count': rescue_before_candidate_count,
        'after_window_candidate_count': rescue_after_candidate_count,
        'representative_before_window': serialize_stage3_window_metadata(representative_before_window),
        'representative_after_window': serialize_stage3_window_metadata(representative_after_window),
        'reconstructed_before_coverage': round(reconstructed_before_coverage, 6),
        'bucket_coverages': serialize_stage3_coverage_summary(rescue_coverage),
        'decision': {
            'screening_result': rescue_screening_result,
            'surviving': rescue_surviving,
            'reason': rescue_reason,
            'bounded_ambiguity': rescue_bounded_ambiguity,
        },
        'cell_traces': build_stage3_rescue_cell_rows(
            candidate_union.union_footprint,
            before_assignments,
            after_assignments,
            rescue_bucket_by_cell,
        ),
    }
    stage3_debug_trace['final_stage3_outcome'] = {
        'screening_result': rescue_screening_result,
        'surviving': rescue_surviving,
        'reason': rescue_reason,
        'mode': 'snapshot_rescue',
        'alignment_mode': 'composite_before_after',
    }

    before_records = [] if representative_before_window is None else get_stage3_window_records(stage3_context, ordered_records,
        int(representative_before_window['window_start']),
        int(representative_before_window['window_end']),
    )
    after_records = [] if representative_after_window is None else get_stage3_window_records(stage3_context, ordered_records,
        int(representative_after_window['window_start']),
        int(representative_after_window['window_end']),
    )
    before_reference_activity = 0.0 if representative_before_window is None else float(representative_before_window['mean_window_activity'])
    after_reference_activity = 0.0 if representative_after_window is None else float(representative_after_window['mean_window_activity'])

    finalize_stage3_debug_trace()
    return build_stage3_screened_union_result(
        candidate_union=candidate_union,
        within_union_records=within_union_records,
        before_records=before_records,
        after_records=after_records,
        mean_movement_strength=mean_movement_strength,
        mean_temporal_persistence=mean_temporal_persistence,
        mean_spatial_extent=mean_spatial_extent,
        before_reference_activity=before_reference_activity,
        after_reference_activity=after_reference_activity,
        screening_result=rescue_screening_result,
        surviving=rescue_surviving,
        reason=rescue_reason,
        reference_windows_reliable=bool(before_assignments) and bool(after_assignments),
        stage3_mode='snapshot_rescue',
        stage3_alignment_mode='composite_before_after',
        resolved_changed_coverage=float(rescue_coverage['resolved_changed_coverage']),
        reconstructed_before_coverage=reconstructed_before_coverage,
        resolved_ambiguous_coverage=float(rescue_coverage['resolved_ambiguous_coverage']),
        before_window=representative_before_window,
        after_window=representative_after_window,
        before_window_candidate_count=rescue_before_candidate_count,
        after_window_candidate_count=rescue_after_candidate_count,
        survived_by_bounded_ambiguity=rescue_bounded_ambiguity,
        stage3_debug_trace=stage3_debug_trace,
    )


def screen_stage3_candidate_unions(
    candidate_unions: Iterable[CandidateUnion],
    records: Iterable[MovementEvidenceRecord],
    sampled_frames: list[dict[str, object]] | None = None,
    chapter_range: ChapterRange | None = None,
    settings: DetectorSettings | None = None,
    use_art_state_prototype: bool = False,
    status_callback: Callable[[str], None] | None = None,
    union_complete_callback: Callable[[list[ScreenedCandidateUnion]], None] | None = None,
) -> list[ScreenedCandidateUnion]:
    ordered_candidate_unions = list(candidate_unions)
    ordered_records = sorted(records, key=lambda record: record.frame_index)
    if sampled_frames is None or chapter_range is None or settings is None:
        return [screen_candidate_union(candidate_union, ordered_records) for candidate_union in ordered_candidate_unions]

    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('OpenCV is required for staged Stage 3 art-state screening.') from exc

    stage3_context = build_stage3_runtime_context(ordered_records, sampled_frames)
    screened_unions: list[ScreenedCandidateUnion] = []
    total_union_count = len(ordered_candidate_unions)
    for candidate_union_index, candidate_union in enumerate(ordered_candidate_unions):
        previous_union_end = None if candidate_union_index == 0 else ordered_candidate_unions[candidate_union_index - 1].end_frame
        next_union_start = None if candidate_union_index == (len(ordered_candidate_unions) - 1) else ordered_candidate_unions[candidate_union_index + 1].start_frame
        union_label = (
            f"Union {candidate_union.union_index} "
            f"({candidate_union.start_time} -> {candidate_union.end_time})"
        )
        union_progress_prefix = (
            f"Runtime Stage 3A - Union {candidate_union_index + 1}/{total_union_count} "
            f"({candidate_union.start_time} -> {candidate_union.end_time})"
        )
        if status_callback is not None:
            status_callback(
                f"Runtime Stage 3A - Screening candidate union {candidate_union_index + 1}/{total_union_count}: {union_label}"
            )
        union_started_at = time.perf_counter()
        screened_union = screen_candidate_union_with_art_state_prototype(
            candidate_union=candidate_union,
            records=ordered_records,
            sampled_frames=sampled_frames,
            chapter_range=chapter_range,
            settings=settings,
            previous_union_end=previous_union_end,
            next_union_start=next_union_start,
            cv2=cv2,
            stage3_context=stage3_context,
            status_callback=status_callback,
            union_progress_prefix=union_progress_prefix,
        )
        screened_unions.append(screened_union)
        if union_complete_callback is not None:
            union_complete_callback(list(screened_unions))
        if status_callback is not None:
            status_callback(
                f"Runtime Stage 3A - Finished candidate union {candidate_union_index + 1}/{total_union_count}: "
                f"{union_label} in {format_stage_elapsed(time.perf_counter() - union_started_at)} "
                f"({screened_union.screening_result})"
            )
    return screened_unions
# ============================================================
# SECTION G - Stage 4 Time Slice Classification
# ============================================================


def build_stage4_time_slice_ranges(
    screened_candidate_union: ScreenedCandidateUnion,
) -> list[tuple[int, int]]:
    union_start = screened_candidate_union.candidate_union.start_frame
    union_end = screened_candidate_union.candidate_union.end_frame
    if union_end <= union_start:
        return []

    def build_probe_window(anchor_frame: int) -> tuple[int, int]:
        window_start = max(union_start, anchor_frame - STAGE4_PROBE_HALF_WINDOW_FRAMES)
        window_end = min(
            union_end,
            window_start + STAGE4_PROBE_LOCAL_WINDOW_FRAMES,
        )
        if window_end <= window_start:
            window_end = min(union_end, window_start + 1)
        return window_start, window_end

    union_duration = union_end - union_start
    anchors: list[int] = []
    if union_duration >= STAGE4_PROBE_INTERVAL_FRAMES:
        anchor_frame = union_start + STAGE4_PROBE_INTERVAL_FRAMES
        while anchor_frame < union_end:
            anchors.append(anchor_frame)
            anchor_frame += STAGE4_PROBE_INTERVAL_FRAMES

    terminal_anchor = max(union_start, union_end - STAGE4_PROBE_TERMINAL_OFFSET_FRAMES)
    if not anchors:
        anchors = [terminal_anchor]
    elif terminal_anchor - anchors[-1] < STAGE4_PROBE_MIN_ANCHOR_SEPARATION_FRAMES:
        anchors[-1] = max(anchors[-1], terminal_anchor)
    else:
        anchors.append(terminal_anchor)

    probe_ranges: list[tuple[int, int]] = []
    for anchor_frame in anchors:
        probe_window = build_probe_window(anchor_frame)
        if not probe_ranges or probe_ranges[-1] != probe_window:
            probe_ranges.append(probe_window)
    return probe_ranges


def count_active_records(records: Iterable[MovementEvidenceRecord]) -> int:
    return sum(1 for record in records if record.movement_present)



def build_canvas_footprint_mask_from_coordinates(
    footprint: Iterable[GridCoordinate],
    canvas_shape: tuple[int, int],
):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('NumPy is required for staged local art-state slice classification.') from exc

    mask = np.zeros(canvas_shape, dtype=np.uint8)
    canvas_height, canvas_width = canvas_shape
    cell_height = max(1, canvas_height // GRID_ROWS)
    cell_width = max(1, canvas_width // GRID_COLUMNS)

    for row_index, column_index in footprint:
        top = row_index * cell_height
        left = column_index * cell_width
        bottom = canvas_height if row_index == (GRID_ROWS - 1) else min(canvas_height, (row_index + 1) * cell_height)
        right = canvas_width if column_index == (GRID_COLUMNS - 1) else min(canvas_width, (column_index + 1) * cell_width)
        mask[top:bottom, left:right] = 255
    return mask



def build_stage4_connected_coordinate_clusters(
    coordinates: Iterable[GridCoordinate],
    adjacency_predicate: Callable[[GridCoordinate, GridCoordinate], bool] | None = None,
) -> list[frozenset[GridCoordinate]]:
    coordinate_set = set(coordinates)
    if not coordinate_set:
        return []

    unvisited = set(coordinate_set)
    clusters: list[frozenset[GridCoordinate]] = []
    while unvisited:
        start_coordinate = min(unvisited)
        pending = [start_coordinate]
        cluster: set[GridCoordinate] = set()
        unvisited.remove(start_coordinate)

        while pending:
            row_index, column_index = pending.pop()
            coordinate = (row_index, column_index)
            cluster.add(coordinate)
            for row_offset in (-1, 0, 1):
                for column_offset in (-1, 0, 1):
                    if row_offset == 0 and column_offset == 0:
                        continue
                    neighbor = (row_index + row_offset, column_index + column_offset)
                    if neighbor not in unvisited:
                        continue
                    if adjacency_predicate is not None and not adjacency_predicate(coordinate, neighbor):
                        continue
                    unvisited.remove(neighbor)
                    pending.append(neighbor)

        clusters.append(frozenset(cluster))

    return sorted(clusters, key=lambda cluster: (min(cluster), len(cluster)))



def build_stage4_slice_local_cell_activity_stats(
    slice_records: Iterable[MovementEvidenceRecord],
) -> dict[GridCoordinate, dict[str, int]]:
    cell_activity_stats: dict[GridCoordinate, dict[str, int]] = {}
    for record in slice_records:
        if not record.touched_grid_coordinates:
            continue
        unique_coordinates = frozenset(record.touched_grid_coordinates)
        for coordinate in unique_coordinates:
            stats = cell_activity_stats.setdefault(
                coordinate,
                {
                    'touch_count': 0,
                    'active_record_count': 0,
                    'first_touch_frame': int(record.frame_index),
                    'last_touch_frame': int(record.frame_index),
                },
            )
            stats['touch_count'] += 1
            if record.movement_present:
                stats['active_record_count'] += 1
            stats['first_touch_frame'] = min(stats['first_touch_frame'], int(record.frame_index))
            stats['last_touch_frame'] = max(stats['last_touch_frame'], int(record.frame_index))
    return cell_activity_stats



def is_stage4_core_active_cell(cell_stats: dict[str, int] | None) -> bool:
    if not cell_stats:
        return False
    return (
        int(cell_stats.get('touch_count', 0)) >= STAGE4_CORE_ACTIVE_CELL_MIN_TOUCH_COUNT
        or int(cell_stats.get('active_record_count', 0)) >= STAGE4_CORE_ACTIVE_CELL_MIN_ACTIVE_RECORDS
    )



def are_stage4_cell_activity_spans_compatible(
    left_stats: dict[str, int] | None,
    right_stats: dict[str, int] | None,
) -> bool:
    if not left_stats or not right_stats:
        return False

    left_start = int(left_stats['first_touch_frame'])
    left_end = int(left_stats['last_touch_frame'])
    right_start = int(right_stats['first_touch_frame'])
    right_end = int(right_stats['last_touch_frame'])

    latest_start = max(left_start, right_start)
    earliest_end = min(left_end, right_end)
    if latest_start <= earliest_end:
        return True

    frame_gap = latest_start - earliest_end
    return frame_gap <= STAGE4_CELL_TIMING_MERGE_TOLERANCE_FRAMES



def split_footprint_into_local_subregions(
    footprint: frozenset[GridCoordinate],
    slice_records: Iterable[MovementEvidenceRecord] | None = None,
) -> list[frozenset[GridCoordinate]]:
    if not footprint:
        return []

    if slice_records is None:
        return build_stage4_connected_coordinate_clusters(footprint)

    cell_activity_stats = build_stage4_slice_local_cell_activity_stats(slice_records)
    core_active_cells = {
        coordinate
        for coordinate in footprint
        if is_stage4_core_active_cell(cell_activity_stats.get(coordinate))
    }

    subregions: list[frozenset[GridCoordinate]] = []
    assigned_coordinates: set[GridCoordinate] = set()

    if core_active_cells:
        def core_cells_are_compatible(left: GridCoordinate, right: GridCoordinate) -> bool:
            return are_stage4_cell_activity_spans_compatible(
                cell_activity_stats.get(left),
                cell_activity_stats.get(right),
            )

        core_clusters = build_stage4_connected_coordinate_clusters(
            core_active_cells,
            adjacency_predicate=core_cells_are_compatible,
        )
        subregions.extend(core_clusters)
        for cluster in core_clusters:
            assigned_coordinates.update(cluster)

    fringe_coordinates = footprint.difference(assigned_coordinates)
    if fringe_coordinates:
        fringe_clusters = build_stage4_connected_coordinate_clusters(fringe_coordinates)
        subregions.extend(fringe_clusters)

    return sorted(subregions, key=lambda cluster: (min(cluster), len(cluster)))



def filter_meaningful_subregions(
    subregions: Iterable[frozenset[GridCoordinate]],
    minimum_cells: int = 2,
) -> list[frozenset[GridCoordinate]]:
    filtered_subregions = [subregion for subregion in subregions if len(subregion) >= minimum_cells]
    if filtered_subregions:
        return filtered_subregions
    return [subregion for subregion in subregions if subregion]



def collect_records_overlapping_subregion(
    records: Iterable[MovementEvidenceRecord],
    subregion: frozenset[GridCoordinate],
) -> list[MovementEvidenceRecord]:
    return [
        record
        for record in records
        if any(coordinate in subregion for coordinate in record.touched_grid_coordinates)
    ]



def compute_local_subregion_window_activity(
    records: Iterable[MovementEvidenceRecord],
    subregion: frozenset[GridCoordinate],
) -> float:
    return average_record_score(
        collect_records_overlapping_subregion(records, subregion),
        'movement_strength_score',
    )



def score_local_subregion_reference_window(
    candidate_window: tuple[int, int],
    records: Iterable[MovementEvidenceRecord],
    sampled_frames: list[dict[str, object]],
    subregion: frozenset[GridCoordinate],
    anchor_frame: int,
) -> dict[str, object] | None:
    window_start, window_end = candidate_window
    window_samples = collect_window_samples(sampled_frames, window_start, window_end)
    if len(window_samples) < STAGE3_ART_STATE_MIN_SAMPLES:
        return None

    window_records = collect_records_in_frame_range(records, window_start, window_end)
    local_window_activity = compute_local_subregion_window_activity(window_records, subregion)
    distance_frames = min(abs(anchor_frame - window_start), abs(anchor_frame - window_end))
    return {
        'window_start': window_start,
        'window_end': window_end,
        'sample_count': len(window_samples),
        'local_window_activity': local_window_activity,
        'distance_frames': distance_frames,
        'trustworthy': local_window_activity <= STAGE4_MAX_REFERENCE_ACTIVITY,
        'tier': 'local' if distance_frames <= STAGE3_ART_STATE_LOCAL_WINDOW_DISTANCE_FRAMES else 'fallback',
    }
def select_local_subregion_reference_window(
    *,
    search_ranges: Iterable[tuple[int, int]],
    anchor_frame: int,
    records: Iterable[MovementEvidenceRecord],
    sampled_frames: list[dict[str, object]],
    subregion: frozenset[GridCoordinate],
    reference_window_frames: int,
    settings: DetectorSettings,
) -> tuple[dict[str, object] | None, int]:
    total_candidate_count = 0

    for search_start, search_end in search_ranges:
        if search_end <= search_start:
            continue
        candidate_windows = build_stage3_reference_window_candidates(
            search_start,
            search_end,
            max(2, reference_window_frames),
            max(settings.sample_stride, max(1, reference_window_frames // 2)),
        )
        total_candidate_count += len(candidate_windows)
        scored_candidates: list[dict[str, object]] = []
        for candidate_window in candidate_windows:
            scored_candidate = score_local_subregion_reference_window(
                candidate_window,
                records,
                sampled_frames,
                subregion,
                anchor_frame,
            )
            if scored_candidate is not None:
                scored_candidates.append(scored_candidate)

        for scored_candidate in sorted(
            scored_candidates,
            key=lambda candidate: (candidate['distance_frames'], candidate['window_start'], candidate['window_end']),
        ):
            if bool(scored_candidate['trustworthy']):
                return scored_candidate, total_candidate_count

    return None, total_candidate_count
def build_slice_reference_search_ranges(
    *,
    slice_start: int,
    slice_end: int,
    parent_start: int,
    parent_end: int,
    sampled_frames: list[dict[str, object]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if not sampled_frames:
        return [], []

    sampled_frame_start = min(int(sample['frame_index']) for sample in sampled_frames)
    sampled_frame_end = max(int(sample['frame_index']) for sample in sampled_frames) + 1
    local_before_start = max(sampled_frame_start, slice_start - STAGE3_ART_STATE_SEARCH_FRAMES)
    local_after_end = min(sampled_frame_end, slice_end + STAGE3_ART_STATE_SEARCH_FRAMES)

    before_ranges = [(local_before_start, slice_start)]
    after_ranges = [(slice_end, local_after_end)]

    if parent_start < local_before_start:
        before_ranges.append((parent_start, slice_start))
    if sampled_frame_start < parent_start:
        before_ranges.append((sampled_frame_start, slice_start))

    if parent_end > local_after_end:
        after_ranges.append((slice_end, parent_end))
    if sampled_frame_end > parent_end:
        after_ranges.append((slice_end, sampled_frame_end))

    return before_ranges, after_ranges



def evaluate_local_subregion_change(
    *,
    subregion: frozenset[GridCoordinate],
    slice_start: int,
    slice_end: int,
    parent_start: int,
    parent_end: int,
    ordered_records: list[MovementEvidenceRecord],
    sampled_frames: list[dict[str, object]],
    settings: DetectorSettings,
    reference_window_frames: int,
    cv2,
) -> dict[str, object]:
    before_ranges, after_ranges = build_slice_reference_search_ranges(
        slice_start=slice_start,
        slice_end=slice_end,
        parent_start=parent_start,
        parent_end=parent_end,
        sampled_frames=sampled_frames,
    )
    selected_before_window, before_candidate_count = select_local_subregion_reference_window(
        search_ranges=before_ranges,
        anchor_frame=slice_start,
        records=ordered_records,
        sampled_frames=sampled_frames,
        subregion=subregion,
        reference_window_frames=reference_window_frames,
        settings=settings,
    )
    selected_after_window, after_candidate_count = select_local_subregion_reference_window(
        search_ranges=after_ranges,
        anchor_frame=slice_end,
        records=ordered_records,
        sampled_frames=sampled_frames,
        subregion=subregion,
        reference_window_frames=reference_window_frames,
        settings=settings,
    )

    before_reference_activity = 0.0 if selected_before_window is None else float(selected_before_window['local_window_activity'])
    after_reference_activity = 0.0 if selected_after_window is None else float(selected_after_window['local_window_activity'])
    if selected_before_window is None or selected_after_window is None:
        return {
            'subregion': subregion,
            'comparison_state': 'reference_missing',
            'settled': False,
            'supported': False,
            'unresolved': True,
            'reference_windows_reliable': False,
            'opening_attribution_start_frame': None,
            'opening_attribution_start_time': None,
            'meaningful_unsettled_activity': True,
            'before_reference_activity': before_reference_activity,
            'after_reference_activity': after_reference_activity,
            'evidence_score': 0.0,
            'persistent_difference_score': 0.0,
            'footprint_support_score': 0.0,
            'after_window_persistence_score': 0.0,
            'before_window_candidate_count': before_candidate_count,
            'after_window_candidate_count': after_candidate_count,
        }

    before_window_start = int(selected_before_window['window_start'])
    before_window_end = int(selected_before_window['window_end'])
    after_window_start = int(selected_after_window['window_start'])
    after_window_end = int(selected_after_window['window_end'])
    before_samples, pre_baseline = build_stage3_art_state_window_baseline(sampled_frames, before_window_start, before_window_end)
    after_samples, post_baseline = build_stage3_art_state_window_baseline(sampled_frames, after_window_start, after_window_end)
    if pre_baseline is None or post_baseline is None or not before_samples or not after_samples:
        return {
            'subregion': subregion,
            'comparison_state': 'reference_missing',
            'settled': False,
            'supported': False,
            'unresolved': True,
            'reference_windows_reliable': False,
            'meaningful_unsettled_activity': True,
            'before_reference_activity': before_reference_activity,
            'after_reference_activity': after_reference_activity,
            'evidence_score': 0.0,
            'persistent_difference_score': 0.0,
            'footprint_support_score': 0.0,
            'after_window_persistence_score': 0.0,
            'before_window_candidate_count': before_candidate_count,
            'after_window_candidate_count': after_candidate_count,
        }

    reference_canvas_shape = get_stage3_canvas_shape(before_samples[0])
    subregion_canvas_mask = build_canvas_footprint_mask_from_coordinates(subregion, reference_canvas_shape)
    subregion_art_state_mask = subregion_canvas_mask
    persistent_mask = build_art_state_change_mask(pre_baseline, post_baseline, settings, cv2)
    focused_persistent_mask = cv2.bitwise_and(persistent_mask, subregion_art_state_mask)
    persistent_difference_score = compute_stage3_art_state_persistent_difference_score(
        focused_persistent_mask,
        len(subregion),
        settings,
    )
    footprint_support_score = compute_stage3_art_state_footprint_support_score(
        focused_persistent_mask,
        len(subregion),
    )
    after_window_persistence_score = compute_stage3_art_state_after_window_persistence_score(
        pre_baseline,
        post_baseline,
        after_samples,
        subregion_art_state_mask,
        len(subregion),
        settings,
        cv2,
    )
    evidence_score = compute_stage3_art_state_evidence_score(
        persistent_difference_score,
        footprint_support_score,
        after_window_persistence_score,
        0.0,
    )
    quiet_after_reference = after_reference_activity <= STAGE4_MAX_REFERENCE_ACTIVITY
    meaningful_persistent_difference = footprint_support_score >= STAGE4_MIN_CONTRAST_SCORE
    settled_changed = (
        quiet_after_reference
        and evidence_score >= STAGE4_VALID_EVIDENCE_SCORE
        and after_window_persistence_score >= 0.45
        and meaningful_persistent_difference
    )
    settled_unchanged = (
        quiet_after_reference
        and evidence_score <= STAGE4_INVALID_EVIDENCE_SCORE
        and not meaningful_persistent_difference
    )
    meaningful_unsettled_activity = (
        not quiet_after_reference
        or (after_window_persistence_score < 0.45 and evidence_score > STAGE4_INVALID_EVIDENCE_SCORE)
    )

    if settled_changed:
        comparison_state = 'settled_changed'
    elif settled_unchanged:
        comparison_state = 'settled_unchanged'
    elif meaningful_unsettled_activity:
        comparison_state = 'unsettled'
    else:
        comparison_state = 'ambiguous'

    return {
        'subregion': subregion,
        'comparison_state': comparison_state,
        'settled': comparison_state in {'settled_changed', 'settled_unchanged'},
        'supported': comparison_state == 'settled_changed',
        'unresolved': comparison_state in {'unsettled', 'ambiguous', 'reference_missing'},
        'reference_windows_reliable': True,
        'meaningful_unsettled_activity': meaningful_unsettled_activity,
        'before_reference_activity': before_reference_activity,
        'after_reference_activity': after_reference_activity,
        'evidence_score': evidence_score,
        'persistent_difference_score': persistent_difference_score,
        'footprint_support_score': footprint_support_score,
        'after_window_persistence_score': after_window_persistence_score,
        'before_window_candidate_count': before_candidate_count,
        'after_window_candidate_count': after_candidate_count,
    }



def summarize_time_slice_local_subregions(
    subregion_results: list[dict[str, object]],
    footprint_size: int,
) -> dict[str, float | int]:
    changed_footprint = sum(
        len(result['subregion'])
        for result in subregion_results
        if result['comparison_state'] == 'settled_changed'
    )
    unchanged_footprint = sum(
        len(result['subregion'])
        for result in subregion_results
        if result['comparison_state'] == 'settled_unchanged'
    )
    unsettled_footprint = sum(
        len(result['subregion'])
        for result in subregion_results
        if result['comparison_state'] == 'unsettled'
    )
    ambiguous_footprint = sum(
        len(result['subregion'])
        for result in subregion_results
        if result['comparison_state'] in {'ambiguous', 'reference_missing'}
    )
    settled_results = [
        result
        for result in subregion_results
        if result['comparison_state'] in {'settled_changed', 'settled_unchanged'}
    ]
    settled_footprint = changed_footprint + unchanged_footprint
    settled_evidence_score = 0.0
    if settled_footprint > 0:
        settled_evidence_score = sum(
            float(result['evidence_score']) * len(result['subregion'])
            for result in settled_results
        ) / settled_footprint

    return {
        'changed_footprint': changed_footprint,
        'unchanged_footprint': unchanged_footprint,
        'unsettled_footprint': unsettled_footprint,
        'ambiguous_footprint': ambiguous_footprint,
        'settled_footprint': settled_footprint,
        'changed_ratio': changed_footprint / max(1, footprint_size),
        'unsettled_ratio': unsettled_footprint / max(1, footprint_size),
        'ambiguous_ratio': ambiguous_footprint / max(1, footprint_size),
        'settled_evidence_score': settled_evidence_score,
    }



def classify_local_time_slice_from_summary(
    summary: dict[str, float | int],
    reference_windows_reliable: bool,
) -> tuple[str, str]:
    changed_ratio = float(summary['changed_ratio'])
    unsettled_ratio = float(summary['unsettled_ratio'])
    ambiguous_ratio = float(summary['ambiguous_ratio'])
    settled_footprint = int(summary['settled_footprint'])
    settled_evidence_score = float(summary['settled_evidence_score'])

    if settled_footprint == 0:
        if unsettled_ratio > 0.0:
            return 'undetermined', 'unsettled_slice_activity'
        if ambiguous_ratio > 0.0 or not reference_windows_reliable:
            return 'undetermined', 'reference_windows_unreliable' if not reference_windows_reliable else 'mixed_slice_evidence'
        return 'invalid', 'weak_slice_activity'

    if (
        changed_ratio >= STAGE4_MIN_SUPPORTED_SUBREGION_FOOTPRINT_RATIO
        and settled_evidence_score >= STAGE4_VALID_EVIDENCE_SCORE
        and unsettled_ratio <= STAGE4_MAX_UNRESOLVED_SUBREGION_FOOTPRINT_RATIO_FOR_VALID
        and ambiguous_ratio <= STAGE4_MAX_UNRESOLVED_SUBREGION_FOOTPRINT_RATIO_FOR_VALID
    ):
        return 'valid', 'slice_activity_supported'

    if changed_ratio < STAGE4_MIN_SUPPORTED_SUBREGION_FOOTPRINT_RATIO:
        if unsettled_ratio > STAGE4_MAX_UNRESOLVED_SUBREGION_FOOTPRINT_RATIO_FOR_VALID:
            return 'undetermined', 'unsettled_slice_activity'
        if ambiguous_ratio > STAGE4_MAX_UNRESOLVED_SUBREGION_FOOTPRINT_RATIO_FOR_VALID or not reference_windows_reliable:
            return 'undetermined', 'reference_windows_unreliable' if not reference_windows_reliable else 'mixed_slice_evidence'
        if settled_evidence_score <= STAGE4_INVALID_EVIDENCE_SCORE:
            return 'invalid', 'weak_slice_activity'
        return 'undetermined', 'mixed_slice_evidence'

    if (
        settled_evidence_score <= STAGE4_INVALID_EVIDENCE_SCORE
        and unsettled_ratio <= STAGE4_MAX_UNRESOLVED_SUBREGION_FOOTPRINT_RATIO_FOR_VALID
        and ambiguous_ratio <= STAGE4_MAX_UNRESOLVED_SUBREGION_FOOTPRINT_RATIO_FOR_VALID
    ):
        return 'invalid', 'weak_slice_activity'

    if unsettled_ratio > 0.0:
        return 'undetermined', 'unsettled_slice_activity'
    if ambiguous_ratio > 0.0 or not reference_windows_reliable:
        return 'undetermined', 'reference_windows_unreliable' if not reference_windows_reliable else 'mixed_slice_evidence'
    return 'undetermined', 'mixed_slice_evidence'
def evaluate_time_slice_local_subregions(
    screened_candidate_union: ScreenedCandidateUnion,
    slice_start: int,
    slice_end: int,
    records: Iterable[MovementEvidenceRecord],
    sampled_frames: list[dict[str, object]] | None,
    settings: DetectorSettings | None,
    reference_window_frames: int,
) -> dict[str, object]:
    ordered_records = sorted(records, key=lambda record: record.frame_index)
    slice_records = collect_records_in_frame_range(ordered_records, slice_start, slice_end)
    footprint = build_footprint_from_records(slice_records)
    footprint_size = len(footprint)
    active_slice_record_count = count_active_records(slice_records)

    fallback_before_records = collect_records_in_frame_range(
        ordered_records,
        max(0, slice_start - reference_window_frames),
        slice_start,
    )
    fallback_after_records = collect_records_in_frame_range(
        ordered_records,
        slice_end,
        slice_end + reference_window_frames,
    )
    fallback_before_activity = average_record_score(fallback_before_records, 'movement_strength_score')
    fallback_after_activity = average_record_score(fallback_after_records, 'movement_strength_score')
    fallback_score = compute_slice_lasting_change_evidence_score(
        slice_records=slice_records,
        footprint_size=footprint_size,
        after_reference_activity=fallback_after_activity,
    )

    if not slice_records or footprint_size == 0:
        return {
            'slice_records': slice_records,
            'footprint': footprint,
            'footprint_size': footprint_size,
            'subregions': [],
            'subregion_results': [],
            'summary': None,
            'classification': 'invalid',
            'reason': 'weak_slice_activity',
            'lasting_change_evidence_score': 0.0,
            'before_reference_activity': fallback_before_activity,
            'after_reference_activity': fallback_after_activity,
            'reference_windows_reliable': True,
            'opening_attribution_start_frame': None,
            'opening_attribution_start_time': None,
        }

    if not sampled_frames or settings is None:
        reference_windows_reliable = (
            len(fallback_before_records) >= STAGE3_MIN_REFERENCE_RECORDS
            and len(fallback_after_records) >= STAGE3_MIN_REFERENCE_RECORDS
        )
        if not reference_windows_reliable:
            classification = 'undetermined'
            reason = 'reference_windows_unreliable'
        elif fallback_score >= STAGE4_VALID_EVIDENCE_SCORE and fallback_after_activity <= STAGE4_MAX_REFERENCE_ACTIVITY:
            classification = 'valid'
            reason = 'slice_activity_supported'
        elif fallback_score <= STAGE4_INVALID_EVIDENCE_SCORE and active_slice_record_count == 0:
            classification = 'invalid'
            reason = 'weak_slice_activity'
        else:
            classification = 'undetermined'
            reason = 'mixed_slice_evidence'
        return {
            'slice_records': slice_records,
            'footprint': footprint,
            'footprint_size': footprint_size,
            'subregions': [],
            'subregion_results': [],
            'summary': None,
            'classification': classification,
            'reason': reason,
            'lasting_change_evidence_score': fallback_score,
            'before_reference_activity': fallback_before_activity,
            'after_reference_activity': fallback_after_activity,
            'reference_windows_reliable': reference_windows_reliable,
        }

    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('OpenCV is required for staged local art-state slice classification.') from exc

    subregions = filter_meaningful_subregions(split_footprint_into_local_subregions(footprint, slice_records))
    subregion_results = [
        evaluate_local_subregion_change(
            subregion=subregion,
            slice_start=slice_start,
            slice_end=slice_end,
            parent_start=screened_candidate_union.candidate_union.start_frame,
            parent_end=screened_candidate_union.candidate_union.end_frame,
            ordered_records=ordered_records,
            sampled_frames=sampled_frames,
            settings=settings,
            reference_window_frames=reference_window_frames,
            cv2=cv2,
        )
        for subregion in subregions
    ]
    if not subregion_results:
        return {
            'slice_records': slice_records,
            'footprint': footprint,
            'footprint_size': footprint_size,
            'subregions': list(subregions),
            'subregion_results': [],
            'summary': None,
            'classification': 'invalid',
            'reason': 'weak_slice_activity',
            'lasting_change_evidence_score': fallback_score,
            'before_reference_activity': fallback_before_activity,
            'after_reference_activity': fallback_after_activity,
            'reference_windows_reliable': False,
        }

    summary = summarize_time_slice_local_subregions(subregion_results, footprint_size)
    reference_windows_reliable = all(bool(result['reference_windows_reliable']) for result in subregion_results)
    classification, reason = classify_local_time_slice_from_summary(summary, reference_windows_reliable)
    average_before_activity = sum(float(result['before_reference_activity']) for result in subregion_results) / len(subregion_results)
    average_after_activity = sum(float(result['after_reference_activity']) for result in subregion_results) / len(subregion_results)

    return {
        'slice_records': slice_records,
        'footprint': footprint,
        'footprint_size': footprint_size,
        'subregions': list(subregions),
        'subregion_results': subregion_results,
        'summary': summary,
        'classification': classification,
        'reason': reason,
        'lasting_change_evidence_score': float(summary['settled_evidence_score']),
        'before_reference_activity': average_before_activity,
        'after_reference_activity': average_after_activity,
        'reference_windows_reliable': reference_windows_reliable,
    }
def build_classified_time_slice_from_evaluation(
    *,
    screened_candidate_union: ScreenedCandidateUnion,
    slice_index: int,
    slice_level: int,
    slice_start: int,
    slice_end: int,
    evaluation: dict[str, object],
) -> ClassifiedTimeSlice:
    return ClassifiedTimeSlice(
        slice_index=slice_index,
        parent_union_index=screened_candidate_union.candidate_union.union_index,
        slice_level=slice_level,
        start_frame=slice_start,
        end_frame=slice_end,
        start_time=Timecode(total_frames=slice_start).to_hhmmssff(),
        end_time=Timecode(total_frames=slice_end).to_hhmmssff(),
        footprint=evaluation['footprint'],
        footprint_size=int(evaluation['footprint_size']),
        within_slice_record_count=len(evaluation['slice_records']),
        classification=str(evaluation['classification']),
        reason=str(evaluation['reason']),
        lasting_change_evidence_score=float(evaluation['lasting_change_evidence_score']),
        before_reference_activity=float(evaluation['before_reference_activity']),
        after_reference_activity=float(evaluation['after_reference_activity']),
        reference_windows_reliable=bool(evaluation['reference_windows_reliable']),
    )



def classify_time_slice_with_evaluation(
    screened_candidate_union: ScreenedCandidateUnion,
    slice_index: int,
    slice_level: int,
    slice_start: int,
    slice_end: int,
    records: Iterable[MovementEvidenceRecord],
    sampled_frames: list[dict[str, object]] | None = None,
    settings: DetectorSettings | None = None,
    reference_window_frames: int = STAGE3_REFERENCE_WINDOW_FRAMES,
) -> tuple[ClassifiedTimeSlice, dict[str, object]]:
    evaluation = evaluate_time_slice_local_subregions(
        screened_candidate_union,
        slice_start,
        slice_end,
        records,
        sampled_frames,
        settings,
        reference_window_frames,
    )
    time_slice = build_classified_time_slice_from_evaluation(
        screened_candidate_union=screened_candidate_union,
        slice_index=slice_index,
        slice_level=slice_level,
        slice_start=slice_start,
        slice_end=slice_end,
        evaluation=evaluation,
    )
    return time_slice, evaluation



def classify_time_slice(
    screened_candidate_union: ScreenedCandidateUnion,
    slice_index: int,
    slice_level: int,
    slice_start: int,
    slice_end: int,
    records: Iterable[MovementEvidenceRecord],
    sampled_frames: list[dict[str, object]] | None = None,
    settings: DetectorSettings | None = None,
    reference_window_frames: int = STAGE3_REFERENCE_WINDOW_FRAMES,
) -> ClassifiedTimeSlice:
    time_slice, _ = classify_time_slice_with_evaluation(
        screened_candidate_union,
        slice_index,
        slice_level,
        slice_start,
        slice_end,
        records,
        sampled_frames,
        settings,
        reference_window_frames,
    )
    return time_slice



def compute_stage4_cell_change_score(
    coordinate: GridCoordinate,
    before_window: dict[str, object],
    current_window: dict[str, object],
    sampled_frames: list[dict[str, object]],
    cell_art_state_masks: dict[GridCoordinate, object],
    baseline_cache: dict[tuple[int, int], tuple[list[dict[str, object]], object | None]],
    comparison_cache: dict[tuple[int, int, int, int], dict[str, object] | None] | None,
    settings: DetectorSettings,
    cv2,
    stage3_context: dict[str, object] | None = None,
) -> float | None:
    details = compute_stage4_cell_change_details(
        coordinate,
        before_window,
        current_window,
        sampled_frames,
        cell_art_state_masks,
        baseline_cache,
        comparison_cache,
        settings,
        cv2,
        stage3_context,
    )
    if details is None:
        return None
    return float(details['change_score'])



def get_stage4_window_pair_comparison(
    before_window: dict[str, object],
    current_window: dict[str, object],
    sampled_frames: list[dict[str, object]],
    baseline_cache: dict[tuple[int, int], tuple[list[dict[str, object]], object | None]],
    comparison_cache: dict[tuple[int, int, int, int], dict[str, object] | None] | None,
    settings: DetectorSettings,
    cv2,
    stage3_context: dict[str, object] | None = None,
) -> dict[str, object] | None:
    cache_key = (
        int(before_window['window_start']),
        int(before_window['window_end']),
        int(current_window['window_start']),
        int(current_window['window_end']),
    )
    if comparison_cache is not None and cache_key in comparison_cache:
        return comparison_cache[cache_key]

    before_samples, before_baseline = get_stage3_window_baseline(
        baseline_cache,
        sampled_frames,
        cache_key[0],
        cache_key[1],
        stage3_context,
    )
    current_samples, current_baseline = get_stage3_window_baseline(
        baseline_cache,
        sampled_frames,
        cache_key[2],
        cache_key[3],
        stage3_context,
    )
    if before_baseline is None or current_baseline is None or not current_samples:
        if comparison_cache is not None:
            comparison_cache[cache_key] = None
        return None

    raw_difference = cv2.absdiff(before_baseline, current_baseline)
    pair_comparison = {
        'before_baseline': before_baseline,
        'current_baseline': current_baseline,
        'raw_difference': raw_difference,
        'threshold_mask': cv2.threshold(
            raw_difference,
            settings.activity_threshold,
            255,
            cv2.THRESH_BINARY,
        )[1],
        'change_mask': build_art_state_change_mask(before_baseline, current_baseline, settings, cv2),
        'before_sample_count': len(before_samples),
        'current_sample_count': len(current_samples),
        'before_window': {
            'window_start': cache_key[0],
            'window_end': cache_key[1],
            'start_time': Timecode(total_frames=cache_key[0]).to_hhmmssff(),
            'end_time': Timecode(total_frames=cache_key[1]).to_hhmmssff(),
            'mean_window_activity': round(float(before_window.get('mean_window_activity', 0.0)), 6),
            'max_window_activity': round(float(before_window.get('max_window_activity', 0.0)), 6),
        },
        'current_window': {
            'window_start': cache_key[2],
            'window_end': cache_key[3],
            'start_time': Timecode(total_frames=cache_key[2]).to_hhmmssff(),
            'end_time': Timecode(total_frames=cache_key[3]).to_hhmmssff(),
            'mean_window_activity': round(float(current_window.get('mean_window_activity', 0.0)), 6),
            'max_window_activity': round(float(current_window.get('max_window_activity', 0.0)), 6),
        },
    }
    if comparison_cache is not None:
        comparison_cache[cache_key] = pair_comparison
    return pair_comparison



def compute_stage4_cell_change_details(
    coordinate: GridCoordinate,
    before_window: dict[str, object],
    current_window: dict[str, object],
    sampled_frames: list[dict[str, object]],
    cell_art_state_masks: dict[GridCoordinate, object],
    baseline_cache: dict[tuple[int, int], tuple[list[dict[str, object]], object | None]],
    comparison_cache: dict[tuple[int, int, int, int], dict[str, object] | None] | None,
    settings: DetectorSettings,
    cv2,
    stage3_context: dict[str, object] | None = None,
) -> dict[str, object] | None:
    pair_comparison = get_stage4_window_pair_comparison(
        before_window,
        current_window,
        sampled_frames,
        baseline_cache,
        comparison_cache,
        settings,
        cv2,
        stage3_context,
    )
    if pair_comparison is None:
        return None

    before_baseline = pair_comparison['before_baseline']
    current_baseline = pair_comparison['current_baseline']
    raw_difference = pair_comparison['raw_difference']
    threshold_mask = pair_comparison['threshold_mask']
    change_mask = pair_comparison['change_mask']
    cell_mask = cell_art_state_masks[coordinate]
    mask_pixels = cell_mask > 0
    mask_pixel_count = int(np.count_nonzero(mask_pixels))
    focused_raw_difference = cv2.bitwise_and(raw_difference, cell_mask)
    focused_threshold_mask = cv2.bitwise_and(threshold_mask, cell_mask)
    focused_change_mask = cv2.bitwise_and(change_mask, cell_mask)
    if mask_pixel_count > 0:
        before_values = before_baseline[mask_pixels]
        current_values = current_baseline[mask_pixels]
        raw_difference_values = focused_raw_difference[mask_pixels]
        threshold_values = focused_threshold_mask[mask_pixels]
        before_baseline_mean = float(before_values.mean())
        current_baseline_mean = float(current_values.mean())
        raw_difference_mean = float(raw_difference_values.mean())
        raw_difference_max = int(raw_difference_values.max())
        raw_difference_nonzero_ratio = float(np.count_nonzero(raw_difference_values)) / float(mask_pixel_count)
        raw_difference_over_threshold_ratio = float(np.count_nonzero(threshold_values)) / float(mask_pixel_count)
    else:
        before_baseline_mean = 0.0
        current_baseline_mean = 0.0
        raw_difference_mean = 0.0
        raw_difference_max = 0
        raw_difference_nonzero_ratio = 0.0
        raw_difference_over_threshold_ratio = 0.0
    focused_change_ratio = compute_binary_mask_ratio(focused_change_mask)
    focused_change_blocks = count_active_blocks(focused_change_mask)
    change_score = compute_stage3_art_state_persistent_difference_score(
        focused_change_mask,
        1,
        settings,
    )
    return {
        'change_score': change_score,
        'focused_change_ratio': focused_change_ratio,
        'focused_change_blocks': focused_change_blocks,
        'mask_pixel_count': mask_pixel_count,
        'before_baseline_mean': round(before_baseline_mean, 6),
        'current_baseline_mean': round(current_baseline_mean, 6),
        'raw_difference_mean': round(raw_difference_mean, 6),
        'raw_difference_max': raw_difference_max,
        'raw_difference_nonzero_ratio': round(raw_difference_nonzero_ratio, 6),
        'raw_difference_over_threshold_ratio': round(raw_difference_over_threshold_ratio, 6),
        'activity_threshold': float(settings.activity_threshold),
        'before_sample_count': int(pair_comparison['before_sample_count']),
        'current_sample_count': int(pair_comparison['current_sample_count']),
        'before_window': pair_comparison['before_window'],
        'current_window': pair_comparison['current_window'],
    }



def build_stage4_probe_cell_rows(coordinates: Iterable[GridCoordinate]) -> list[dict[str, object]]:
    return [
        {
            'coordinate': list(coordinate),
            'label': format_grid_coordinate_label(coordinate),
            'details': serialize_grid_coordinate(coordinate),
        }
        for coordinate in sorted(coordinates)
    ]



def stage4_coordinate_sets_have_local_connection(
    first: Iterable[GridCoordinate],
    second: Iterable[GridCoordinate],
) -> bool:
    first_set = frozenset(first)
    second_set = frozenset(second)
    if not first_set or not second_set:
        return False
    if first_set.intersection(second_set):
        return True
    return any(
        abs(first_row - second_row) <= 1 and abs(first_column - second_column) <= 1
        for first_row, first_column in first_set
        for second_row, second_column in second_set
    )



def stage4_collect_locally_connected_coordinates(
    base_coordinates: Iterable[GridCoordinate],
    candidate_coordinates: Iterable[GridCoordinate],
) -> frozenset[GridCoordinate]:
    base_set = frozenset(base_coordinates)
    candidate_set = frozenset(candidate_coordinates)
    if not base_set or not candidate_set:
        return frozenset()
    return frozenset(
        coordinate
        for coordinate in candidate_set
        if any(
            abs(coordinate[0] - base_coordinate[0]) <= 1
            and abs(coordinate[1] - base_coordinate[1]) <= 1
            for base_coordinate in base_set
        )
    )



def build_stage4_probe_reason_summary(
    *,
    probe_label: str,
    changed_cell_count: int,
    judgeable_cell_count: int,
    unresolved_cell_count: int,
    judgeable_coverage: float,
    structural_holding_support: bool,
    opening_zone_low_confidence: bool,
    late_resolution_only: bool,
) -> str:
    opening_note = ' opening-zone-low-confidence;' if opening_zone_low_confidence else ''
    if probe_label == 'positive':
        return (
            f'positive: {changed_cell_count} confirmed changed cells; '
            f'{judgeable_cell_count} judgeable; {unresolved_cell_count} unresolved;'
            f'{opening_note}'
        ).strip()
    if probe_label == 'negative':
        return (
            f'negative: no confirmed changed cells; '
            f'judgeable coverage {judgeable_coverage:.2f}; '
            f'{unresolved_cell_count} unresolved; no structural holding support;'
            f'{opening_note}'
        ).strip()
    if probe_label == 'holding':
        late_resolution_note = ' late-resolution-only;' if late_resolution_only else ''
        return (
            f'holding: unresolved continuity remains plausible; '
            f'{unresolved_cell_count} unresolved; '
            f'structural_holding_support={structural_holding_support};'
            f'{late_resolution_note}'
            f'{opening_note}'
        ).strip()
    return (
        f'unclear: unresolved evidence is present but too weak to keep the band alive; '
        f'{unresolved_cell_count} unresolved; '
        f'judgeable coverage {judgeable_coverage:.2f};'
        f'{opening_note}'
    ).strip()



def find_stage4_opening_attribution_start_frame(
    *,
    screened_candidate_union: ScreenedCandidateUnion,
    ordered_records: list[MovementEvidenceRecord],
    search_start_frame: int,
    slice_end: int,
    target_coordinates: frozenset[GridCoordinate],
    max_gap_frames: int = 4,
) -> int | None:
    if not target_coordinates:
        return None

    relevant_records = [
        record
        for record in collect_records_in_frame_range(
            ordered_records,
            search_start_frame,
            slice_end,
        )
        if record.movement_present and any(
            coordinate in target_coordinates
            for coordinate in record.touched_grid_coordinates
        )
    ]
    if not relevant_records:
        return None

    earliest_frame = int(relevant_records[-1].frame_index)
    previous_frame = earliest_frame
    for record in reversed(relevant_records[:-1]):
        frame_index = int(record.frame_index)
        if previous_frame - frame_index > max_gap_frames:
            break
        earliest_frame = frame_index
        previous_frame = frame_index
    return max(search_start_frame, earliest_frame)


def stage4_probe_is_near_global_opening_candidate(evaluation: dict[str, object]) -> bool:
    if str(evaluation.get('probe_label', 'unclear')) != 'positive':
        return False

    changed_support_score = float(evaluation.get('judgeable_changed_support_score', 0.0))
    unresolved_cell_count = int(evaluation.get('unresolved_cell_count', 0))
    unchanged_cell_count = int(evaluation.get('confirmed_unchanged_cell_count', 0))
    changed_cells = extract_stage4_probe_coordinates(evaluation.get('confirmed_changed_cells', []))
    return (
        bool(changed_cells)
        and changed_support_score >= STAGE4_NEAR_GLOBAL_OPENING_MIN_CHANGED_SUPPORT
        and unresolved_cell_count <= STAGE4_NEAR_GLOBAL_OPENING_MAX_UNRESOLVED_CELLS
        and unchanged_cell_count <= STAGE4_NEAR_GLOBAL_OPENING_MAX_UNCHANGED_CELLS
        and stage4_coordinate_span_is_broad(changed_cells)
    )


def find_stage4_near_global_opening_attribution_start_frame(
    *,
    screened_candidate_union: ScreenedCandidateUnion,
    ordered_records: list[MovementEvidenceRecord],
    search_start_frame: int,
    slice_end: int,
    target_coordinates: frozenset[GridCoordinate],
) -> int | None:
    if not target_coordinates:
        return None

    required_touch_count = max(
        1,
        int(np.ceil(len(target_coordinates) * STAGE4_NEAR_GLOBAL_OPENING_MIN_TOUCHED_COVERAGE)),
    )
    qualifying_records = [
        record
        for record in collect_records_in_frame_range(
            ordered_records,
            search_start_frame,
            slice_end,
        )
        if record.movement_present
        and len(set(record.touched_grid_coordinates).intersection(target_coordinates)) >= required_touch_count
        and stage4_coordinate_span_is_broad(
            frozenset(set(record.touched_grid_coordinates).intersection(target_coordinates))
        )
    ]
    if not qualifying_records:
        return None

    return max(search_start_frame, int(qualifying_records[0].frame_index))


def find_stage4_changed_frontier_opening_attribution_start_frame(
    *,
    screened_candidate_union: ScreenedCandidateUnion,
    ordered_records: list[MovementEvidenceRecord],
    search_start_frame: int,
    slice_end: int,
    target_coordinates: frozenset[GridCoordinate],
    max_gap_frames: int = 4,
) -> int | None:
    if not target_coordinates:
        return None

    required_touch_count = 1
    if len(target_coordinates) > 1:
        required_touch_count = max(
            STAGE4_CHANGED_FRONTIER_OPENING_MIN_TOUCHED_CELLS,
            int(np.ceil(len(target_coordinates) * STAGE4_CHANGED_FRONTIER_OPENING_MIN_TOUCHED_COVERAGE)),
        )

    qualifying_records = [
        record
        for record in collect_records_in_frame_range(
            ordered_records,
            search_start_frame,
            slice_end,
        )
        if record.movement_present
        and len(set(record.touched_grid_coordinates).intersection(target_coordinates)) >= required_touch_count
    ]
    if not qualifying_records:
        return None

    earliest_frame = int(qualifying_records[-1].frame_index)
    previous_frame = earliest_frame
    for record in reversed(qualifying_records[:-1]):
        frame_index = int(record.frame_index)
        if previous_frame - frame_index > max_gap_frames:
            break
        earliest_frame = frame_index
        previous_frame = frame_index
    return max(search_start_frame, earliest_frame)


def evaluate_stage4_cell_level_probe(
    *,
    screened_candidate_union: ScreenedCandidateUnion,
    slice_start: int,
    slice_end: int,
    ordered_records: list[MovementEvidenceRecord],
    sampled_frames: list[dict[str, object]] | None,
    settings: DetectorSettings | None,
    carried_unresolved_cells: frozenset[GridCoordinate],
    carried_recently_changed_cells: frozenset[GridCoordinate],
    baseline_comparison_state: dict[GridCoordinate, dict[str, object]],
    previous_probe_end: int | None,
    previous_probe_blocks_opening_attribution: bool = False,
    carried_post_reset_contaminated_cells: frozenset[GridCoordinate] = frozenset(),
    reference_window_frames: int = STAGE3_REFERENCE_WINDOW_FRAMES,
) -> dict[str, object]:
    probe_started_at = time.perf_counter()
    probe_timings: dict[str, float] = {}

    def record_probe_timing(label: str, started_at: float) -> None:
        probe_timings[label] = round(time.perf_counter() - started_at, 6)

    def finalize_probe_evaluation(payload: dict[str, object]) -> dict[str, object]:
        probe_timings['total_probe_evaluation_seconds'] = round(time.perf_counter() - probe_started_at, 6)
        payload['probe_timings'] = dict(probe_timings)
        return payload

    setup_started_at = time.perf_counter()
    slice_records = collect_records_in_frame_range(ordered_records, slice_start, slice_end)
    relevant_record_start = (
        screened_candidate_union.candidate_union.start_frame
        if previous_probe_end is None
        else min(previous_probe_end, slice_start)
    )
    relevant_source_records = collect_records_in_frame_range(ordered_records, relevant_record_start, slice_end)
    freshly_touched_cells = frozenset(
        coordinate
        for record in relevant_source_records
        if record.movement_present
        for coordinate in record.touched_grid_coordinates
        if coordinate in screened_candidate_union.candidate_union.union_footprint
    )
    recently_active_cells = frozenset(
        freshly_touched_cells
        .union(carried_unresolved_cells)
        .union(carried_recently_changed_cells)
    )
    opening_zone_low_confidence = previous_probe_end is None and slice_start > screened_candidate_union.candidate_union.start_frame

    fallback_score = compute_slice_lasting_change_evidence_score(
        slice_records=slice_records,
        footprint_size=len(recently_active_cells),
        after_reference_activity=0.0,
    )
    if not recently_active_cells:
        record_probe_timing('setup_seconds', setup_started_at)
        return finalize_probe_evaluation({
            'slice_records': slice_records,
            'footprint': recently_active_cells,
            'footprint_size': 0,
            'subregions': [],
            'subregion_results': [],
            'summary': None,
            'probe_model': 'cell_level',
            'probe_anchor_time': Timecode(total_frames=slice_end).to_hhmmssff(),
            'probe_local_evaluation_window': {
                'start_frame': slice_start,
                'end_frame': slice_end,
                'start_time': Timecode(total_frames=slice_start).to_hhmmssff(),
                'end_time': Timecode(total_frames=slice_end).to_hhmmssff(),
            },
            'recently_active_cells': [],
            'judgeable_cells': [],
            'contaminated_cells': [],
            'confirmed_changed_cells': [],
            'confirmed_unchanged_cells': [],
            'unresolved_cells': [],
            'cell_results': [],
            'changed_support_score': 0.0,
            'judgeable_changed_support_score': 0.0,
            'unchanged_support_score': 0.0,
            'unresolved_support_score': 0.0,
            'relevant_cell_count': 0,
            'judgeable_cell_count': 0,
            'confirmed_changed_cell_count': 0,
            'confirmed_unchanged_cell_count': 0,
            'unresolved_cell_count': 0,
            'judgeable_coverage': 1.0,
            'structural_holding_support': False,
            'opening_zone_low_confidence': opening_zone_low_confidence,
            'classification': 'invalid',
            'reason': 'probe_cell_no_relevant_activity',
            'reason_summary': 'negative: no relevant cells in this probe window',
            'lasting_change_evidence_score': 0.0,
            'before_reference_activity': 0.0,
            'after_reference_activity': 0.0,
            'reference_windows_reliable': True,
            'probe_label': 'negative',
        })

    if not sampled_frames or settings is None:
        record_probe_timing('setup_seconds', setup_started_at)
        probe_label = 'holding' if freshly_touched_cells or carried_unresolved_cells else 'unclear'
        return finalize_probe_evaluation({
            'slice_records': slice_records,
            'footprint': recently_active_cells,
            'footprint_size': len(recently_active_cells),
            'subregions': [],
            'subregion_results': [],
            'summary': None,
            'probe_model': 'cell_level',
            'probe_anchor_time': Timecode(total_frames=slice_end).to_hhmmssff(),
            'probe_local_evaluation_window': {
                'start_frame': slice_start,
                'end_frame': slice_end,
                'start_time': Timecode(total_frames=slice_start).to_hhmmssff(),
                'end_time': Timecode(total_frames=slice_end).to_hhmmssff(),
            },
            'recently_active_cells': build_stage4_probe_cell_rows(recently_active_cells),
            'judgeable_cells': [],
            'contaminated_cells': build_stage4_probe_cell_rows(recently_active_cells),
            'confirmed_changed_cells': [],
            'confirmed_unchanged_cells': [],
            'unresolved_cells': build_stage4_probe_cell_rows(recently_active_cells),
            'cell_results': [],
            'changed_support_score': 0.0,
            'judgeable_changed_support_score': 0.0,
            'unchanged_support_score': 0.0,
            'unresolved_support_score': 1.0,
            'relevant_cell_count': len(recently_active_cells),
            'judgeable_cell_count': 0,
            'confirmed_changed_cell_count': 0,
            'confirmed_unchanged_cell_count': 0,
            'unresolved_cell_count': len(recently_active_cells),
            'judgeable_coverage': 0.0,
            'structural_holding_support': bool(freshly_touched_cells or carried_unresolved_cells),
            'opening_zone_low_confidence': opening_zone_low_confidence,
            'classification': 'undetermined',
            'reason': 'probe_references_unavailable',
            'reason_summary': 'holding: probe references unavailable for relevant cells',
            'lasting_change_evidence_score': fallback_score,
            'before_reference_activity': 0.0,
            'after_reference_activity': 0.0,
            'reference_windows_reliable': False,
            'probe_label': probe_label,
        })

    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('OpenCV is required for staged cell-level probe classification.') from exc

    sampled_frame_start = min(int(sample['frame_index']) for sample in sampled_frames)
    sampled_frame_end = max(int(sample['frame_index']) for sample in sampled_frames) + 1
    post_union_resolution_end = min(
        sampled_frame_end,
        screened_candidate_union.candidate_union.end_frame + STAGE4_POST_UNION_LATE_RESOLUTION_FRAMES,
    )
    reference_canvas_shape = get_stage3_canvas_shape(sampled_frames[0])
    cell_art_state_masks = build_stage3_cell_art_state_masks(recently_active_cells, reference_canvas_shape)
    stage3_context = build_stage3_runtime_context(ordered_records, sampled_frames)
    baseline_cache: dict[tuple[int, int], tuple[list[dict[str, object]], object | None]] = {}
    comparison_cache: dict[tuple[int, int, int, int], dict[str, object] | None] = {}
    record_probe_timing('setup_seconds', setup_started_at)
    before_reference_assignment_started_at = time.perf_counter()

    baseline_assigned_coordinates = frozenset(
        coordinate
        for coordinate in recently_active_cells
        if coordinate in baseline_comparison_state
    )
    unresolved_baseline_cells = recently_active_cells.difference(baseline_assigned_coordinates)
    searched_before_assignments: dict[GridCoordinate, dict[str, object]] = {}
    before_candidate_count = 0
    if unresolved_baseline_cells:
        searched_before_assignments, before_candidate_count = select_stage3_cell_reference_windows(
            search_start=max(sampled_frame_start, screened_candidate_union.candidate_union.start_frame - STAGE3_ART_STATE_RESCUE_SEARCH_FRAMES),
            search_end=max(sampled_frame_start, slice_start),
            prefer_nearest=True,
            footprint=recently_active_cells,
            endpoint_coordinates=frozenset(),
            require_external_movement_for_all_cells=False,
            require_external_movement_for_endpoints=False,
            require_external_movement_outside_coordinate_for_all_cells=False,
            ordered_records=ordered_records,
            sampled_frames=sampled_frames,
            cell_art_state_masks=cell_art_state_masks,
            settings=settings,
            cv2=cv2,
            stage3_context=stage3_context,
            target_coordinates=unresolved_baseline_cells,
        )
    before_assignments: dict[GridCoordinate, dict[str, object]] = {
        coordinate: baseline_comparison_state[coordinate]
        for coordinate in baseline_assigned_coordinates
    }
    before_assignments.update(searched_before_assignments)
    record_probe_timing('before_reference_assignment_seconds', before_reference_assignment_started_at)

    current_window_touched_cells = frozenset(
        coordinate
        for record in slice_records
        if record.movement_present
        for coordinate in record.touched_grid_coordinates
        if coordinate in recently_active_cells
    )
    post_reset_contaminated_cells = frozenset(
        coordinate
        for coordinate in carried_post_reset_contaminated_cells
        if coordinate in recently_active_cells
    )


    active_cell_last_frames: dict[GridCoordinate, int] = {}
    for record in relevant_source_records:
        if not record.movement_present:
            continue
        for coordinate in record.touched_grid_coordinates:
            if coordinate in recently_active_cells:
                active_cell_last_frames[coordinate] = max(
                    active_cell_last_frames.get(coordinate, slice_start),
                    int(record.frame_index),
                )

    current_reference_assignment_started_at = time.perf_counter()
    current_assignments: dict[GridCoordinate, dict[str, object]] = {}
    current_candidate_count = 0
    current_search_starts: list[int] = []
    current_search_ends: list[int] = []
    current_search_groups: dict[tuple[int, int, bool], set[GridCoordinate]] = {}
    for coordinate in sorted(recently_active_cells):
        coordinate_had_recent_activity = coordinate in active_cell_last_frames
        if coordinate_had_recent_activity:
            coordinate_search_start = min(
                post_union_resolution_end,
                active_cell_last_frames[coordinate] + 1,
            )
        else:
            coordinate_search_start = max(
                sampled_frame_start,
                slice_start - STAGE4_PROBE_HALF_WINDOW_FRAMES,
            )
        coordinate_search_end = min(
            post_union_resolution_end,
            max(
                coordinate_search_start + STAGE4_PROBE_CURRENT_SEARCH_FRAMES,
                slice_end + STAGE4_PROBE_HALF_WINDOW_FRAMES,
            ),
        )
        if coordinate in post_reset_contaminated_cells:
            coordinate_search_end = min(
                coordinate_search_end,
                slice_end + STAGE4_POST_RESET_LOCAL_SEARCH_FRAMES,
            )
        current_search_starts.append(coordinate_search_start)
        current_search_ends.append(coordinate_search_end)
        if coordinate_search_end <= coordinate_search_start:
            continue
        group_key = (
            coordinate_search_start,
            coordinate_search_end,
            (not coordinate_had_recent_activity),
        )
        current_search_groups.setdefault(group_key, set()).add(coordinate)

    for (
        coordinate_search_start,
        coordinate_search_end,
        require_external_movement_outside_coordinate,
    ), grouped_coordinates in sorted(current_search_groups.items(), key=lambda item: item[0]):
        grouped_assignments, grouped_candidate_count = select_stage3_cell_reference_windows(
            search_start=coordinate_search_start,
            search_end=coordinate_search_end,
            prefer_nearest=False,
            footprint=recently_active_cells,
            endpoint_coordinates=frozenset(),
            require_external_movement_for_all_cells=False,
            require_external_movement_for_endpoints=False,
            require_external_movement_outside_coordinate_for_all_cells=require_external_movement_outside_coordinate,
            ordered_records=ordered_records,
            sampled_frames=sampled_frames,
            cell_art_state_masks=cell_art_state_masks,
            settings=settings,
            cv2=cv2,
            stage3_context=stage3_context,
            target_coordinates=frozenset(grouped_coordinates),
        )
        current_candidate_count += grouped_candidate_count
        current_assignments.update(grouped_assignments)
    record_probe_timing('current_reference_assignment_seconds', current_reference_assignment_started_at)
    current_search_start = min(current_search_starts) if current_search_starts else min(post_union_resolution_end, slice_end)
    current_search_end = max(current_search_ends) if current_search_ends else min(post_union_resolution_end, slice_end)
    judgeable_cells = frozenset(
        coordinate
        for coordinate in recently_active_cells
        if coordinate in before_assignments and coordinate in current_assignments
    )
    contaminated_cells = frozenset(current_window_touched_cells.difference(judgeable_cells))

    cell_comparison_started_at = time.perf_counter()
    confirmed_changed_cells: set[GridCoordinate] = set()
    confirmed_unchanged_cells: set[GridCoordinate] = set()
    unresolved_cells: set[GridCoordinate] = set(recently_active_cells.difference(judgeable_cells))
    cell_results: list[dict[str, object]] = []
    cell_reference_debug: list[dict[str, object]] = []

    for coordinate in sorted(judgeable_cells):
        before_window = before_assignments.get(coordinate)
        current_window = current_assignments.get(coordinate)
        if before_window is None or current_window is None:
            unresolved_cells.add(coordinate)
            cell_results.append({
                'coordinate': list(coordinate),
                'label': format_grid_coordinate_label(coordinate),
                'classification': 'unresolved',
                'change_score': None,
                'reason': 'reference_unavailable',
            })
            cell_reference_debug.append({
                'coordinate': list(coordinate),
                'label': format_grid_coordinate_label(coordinate),
                'classification': 'unresolved',
                'change_score': None,
                'focused_change_ratio': None,
                'focused_change_blocks': None,
                'mask_pixel_count': None,
                'before_baseline_mean': None,
                'current_baseline_mean': None,
                'raw_difference_mean': None,
                'raw_difference_max': None,
                'raw_difference_nonzero_ratio': None,
                'raw_difference_over_threshold_ratio': None,
                'activity_threshold': float(settings.activity_threshold),
                'before_window_source': 'inherited_baseline' if coordinate in baseline_assigned_coordinates else 'searched_before_window',
                'before_window': None,
                'current_window_source': 'searched_current_window',
                'current_window': None,
                'before_sample_count': 0,
                'current_sample_count': 0,
                'reason': 'reference_unavailable',
            })
            continue

        change_details = compute_stage4_cell_change_details(
            coordinate,
            before_window,
            current_window,
            sampled_frames,
            cell_art_state_masks,
            baseline_cache,
            comparison_cache,
            settings,
            cv2,
            stage3_context,
        )
        change_score = None if change_details is None else float(change_details['change_score'])
        coordinate_is_post_reset_contaminated = (
            coordinate in post_reset_contaminated_cells
            and coordinate in current_window_touched_cells
        )
        if change_score is None or coordinate_is_post_reset_contaminated:
            unresolved_cells.add(coordinate)
            cell_classification = 'unresolved'
        elif change_score >= 0.50:
            confirmed_changed_cells.add(coordinate)
            cell_classification = 'confirmed_changed'
        elif change_score <= 0.20:
            confirmed_unchanged_cells.add(coordinate)
            cell_classification = 'confirmed_unchanged'
        else:
            unresolved_cells.add(coordinate)
            cell_classification = 'unresolved'
        cell_results.append({
            'coordinate': list(coordinate),
            'label': format_grid_coordinate_label(coordinate),
            'classification': cell_classification,
            'change_score': None if change_score is None else round(change_score, 6),
            'reason': 'cell_level_probe_comparison',
        })
        cell_reference_debug.append({
            'coordinate': list(coordinate),
            'label': format_grid_coordinate_label(coordinate),
            'classification': cell_classification,
            'change_score': None if change_score is None else round(change_score, 6),
            'focused_change_ratio': None if change_details is None else round(float(change_details['focused_change_ratio']), 6),
            'focused_change_blocks': None if change_details is None else int(change_details['focused_change_blocks']),
            'mask_pixel_count': None if change_details is None else int(change_details['mask_pixel_count']),
            'before_baseline_mean': None if change_details is None else round(float(change_details['before_baseline_mean']), 6),
            'current_baseline_mean': None if change_details is None else round(float(change_details['current_baseline_mean']), 6),
            'raw_difference_mean': None if change_details is None else round(float(change_details['raw_difference_mean']), 6),
            'raw_difference_max': None if change_details is None else int(change_details['raw_difference_max']),
            'raw_difference_nonzero_ratio': None if change_details is None else round(float(change_details['raw_difference_nonzero_ratio']), 6),
            'raw_difference_over_threshold_ratio': None if change_details is None else round(float(change_details['raw_difference_over_threshold_ratio']), 6),
            'activity_threshold': None if change_details is None else float(change_details['activity_threshold']),
            'before_window_source': 'inherited_baseline' if coordinate in baseline_assigned_coordinates else 'searched_before_window',
            'before_window': None if change_details is None else change_details['before_window'],
            'current_window_source': 'searched_current_window',
            'current_window': None if change_details is None else change_details['current_window'],
            'before_sample_count': 0 if change_details is None else int(change_details['before_sample_count']),
            'current_sample_count': 0 if change_details is None else int(change_details['current_sample_count']),
            'reason': 'cell_level_probe_comparison',
        })

    record_probe_timing('cell_comparison_seconds', cell_comparison_started_at)

    classification_started_at = time.perf_counter()
    current_window_activities = [
        float(window.get('mean_window_activity', 0.0))
        for window in current_assignments.values()
        if isinstance(window, dict)
    ]
    after_reference_activity = (
        sum(current_window_activities) / len(current_window_activities)
        if current_window_activities
        else 0.0
    )
    total_relevant_cells = max(1, len(recently_active_cells))
    total_judgeable_cells = max(1, len(judgeable_cells))
    effective_unresolved_cells = unresolved_cells.difference(post_reset_contaminated_cells)
    changed_support_score = len(confirmed_changed_cells) / total_relevant_cells
    unchanged_support_score = len(confirmed_unchanged_cells) / total_relevant_cells
    unresolved_support_score = len(unresolved_cells) / total_relevant_cells
    effective_unresolved_support_score = len(effective_unresolved_cells) / total_relevant_cells
    judgeable_changed_support_score = len(confirmed_changed_cells) / total_judgeable_cells
    judgeable_coverage = len(judgeable_cells) / total_relevant_cells
    changed_cell_count = len(confirmed_changed_cells)
    unresolved_cell_count = len(unresolved_cells)
    effective_unresolved_cell_count = len(effective_unresolved_cells)
    changed_has_fresh_probe_support = stage4_coordinate_sets_have_local_connection(
        confirmed_changed_cells,
        current_window_touched_cells,
    )
    changed_touch_frame_count = sum(
        1
        for record in slice_records
        if record.movement_present and any(
            coordinate in confirmed_changed_cells
            for coordinate in record.touched_grid_coordinates
        )
    )
    changed_touch_frame_ratio = changed_touch_frame_count / max(1, slice_end - slice_start)
    changed_has_carried_resolution_support = stage4_coordinate_sets_have_local_connection(
        confirmed_changed_cells,
        carried_unresolved_cells,
    )
    late_resolution_only = (
        changed_cell_count > 0
        and changed_has_carried_resolution_support
        and changed_touch_frame_count <= STAGE4_PROBE_LATE_RESOLUTION_MAX_CHANGED_TOUCH_FRAMES
        and changed_touch_frame_ratio <= STAGE4_PROBE_LATE_RESOLUTION_MAX_CHANGED_TOUCH_RATIO
        and after_reference_activity <= STAGE4_PROBE_LATE_RESOLUTION_MAX_AFTER_ACTIVITY
    )

    unresolved_has_recent_activity_continuity = stage4_coordinate_sets_have_local_connection(effective_unresolved_cells, freshly_touched_cells)
    unresolved_has_carried_continuity = stage4_coordinate_sets_have_local_connection(effective_unresolved_cells, carried_unresolved_cells)
    structural_holding_support = bool(effective_unresolved_cells) and (
        unresolved_has_recent_activity_continuity
        or unresolved_has_carried_continuity
    )
    negative_has_broad_coverage = judgeable_coverage >= STAGE4_PROBE_NEGATIVE_MIN_JUDGEABLE_COVERAGE
    negative_has_tiny_unresolved_remainder = effective_unresolved_support_score <= STAGE4_PROBE_NEGATIVE_MAX_UNRESOLVED_SUPPORT

    if changed_cell_count >= STAGE4_PROBE_MIN_ABSOLUTE_CHANGED_CELLS and not late_resolution_only:
        classification = 'valid'
        reason = 'probe_cell_changed_support'
        probe_label = 'positive'
    elif late_resolution_only:
        classification = 'undetermined'
        reason = 'probe_cell_late_resolution_support'
        probe_label = 'holding'
    elif (
        changed_cell_count == 0
        and negative_has_broad_coverage
        and negative_has_tiny_unresolved_remainder
        and not structural_holding_support
    ):
        classification = 'invalid'
        reason = 'probe_cell_broad_unchanged_support'
        probe_label = 'negative'
    elif (
        changed_cell_count == 0
        and unresolved_support_score >= STAGE4_PROBE_HOLDING_MIN_UNRESOLVED_SUPPORT
        and structural_holding_support
    ):
        classification = 'undetermined'
        reason = 'probe_cell_structural_holding_support'
        probe_label = 'holding'
    else:
        classification = 'undetermined'
        reason = 'probe_cell_ambiguous_support'
        probe_label = 'unclear'

    opening_attribution_start_frame: int | None = None
    contamination_only_gap = (
        bool(carried_post_reset_contaminated_cells)
        and bool(carried_unresolved_cells)
        and frozenset(carried_unresolved_cells).issubset(carried_post_reset_contaminated_cells)
    )
    near_global_opening_candidate = stage4_probe_is_near_global_opening_candidate({
        'probe_label': probe_label,
        'judgeable_changed_support_score': judgeable_changed_support_score,
        'unresolved_cell_count': unresolved_cell_count,
        'confirmed_unchanged_cell_count': len(confirmed_unchanged_cells),
        'confirmed_changed_cells': build_stage4_probe_cell_rows(confirmed_changed_cells),
    })
    supports_band_opening_attribution = (
        probe_label == 'positive'
        and changed_touch_frame_count > 0
        and (
            not previous_probe_blocks_opening_attribution
            or near_global_opening_candidate
        )
        and (
            opening_zone_low_confidence
            or (
                previous_probe_end is not None
                and not carried_recently_changed_cells
                and (
                    not carried_unresolved_cells
                    or contamination_only_gap
                )
            )
        )
    )
    supports_changed_frontier_opening_attribution = (
        probe_label == 'positive'
        and previous_probe_blocks_opening_attribution
        and previous_probe_end is not None
        and changed_cell_count > 0
        and not near_global_opening_candidate
        and not opening_zone_low_confidence
        and not carried_recently_changed_cells
        and not carried_unresolved_cells
        and changed_touch_frame_count > 0
    )
    opening_search_start_frame = (
        screened_candidate_union.candidate_union.start_frame
        if previous_probe_end is None
        else previous_probe_end
    )
    if supports_band_opening_attribution:
        if near_global_opening_candidate:
            opening_attribution_start_frame = find_stage4_near_global_opening_attribution_start_frame(
                screened_candidate_union=screened_candidate_union,
                ordered_records=ordered_records,
                search_start_frame=opening_search_start_frame,
                slice_end=slice_end,
                target_coordinates=recently_active_cells,
            )
        if opening_attribution_start_frame is None:
            opening_attribution_start_frame = find_stage4_opening_attribution_start_frame(
                screened_candidate_union=screened_candidate_union,
                ordered_records=ordered_records,
                search_start_frame=opening_search_start_frame,
                slice_end=slice_end,
                target_coordinates=recently_active_cells,
            )
    elif supports_changed_frontier_opening_attribution:
        opening_attribution_start_frame = find_stage4_changed_frontier_opening_attribution_start_frame(
            screened_candidate_union=screened_candidate_union,
            ordered_records=ordered_records,
            search_start_frame=opening_search_start_frame,
            slice_end=slice_end,
            target_coordinates=confirmed_changed_cells,
        )

    confirmed_changed_cell_rows = build_stage4_probe_cell_rows(confirmed_changed_cells)
    confirmed_unchanged_cell_rows = build_stage4_probe_cell_rows(confirmed_unchanged_cells)
    unresolved_cell_rows = build_stage4_probe_cell_rows(unresolved_cells)
    state_reset_candidate = stage4_probe_is_state_reset_candidate({
        'probe_label': probe_label,
        'unresolved_cell_count': unresolved_cell_count,
        'confirmed_changed_cell_count': changed_cell_count,
        'changed_support_score': changed_support_score,
        'confirmed_changed_cells': confirmed_changed_cell_rows,
    })
    post_reset_contaminated_carry = (
        stage4_extract_post_reset_contaminated_cells({
            'probe_label': probe_label,
            'unresolved_cell_count': unresolved_cell_count,
            'confirmed_changed_cell_count': changed_cell_count,
            'changed_support_score': changed_support_score,
            'confirmed_changed_cells': confirmed_changed_cell_rows,
            'unresolved_cells': unresolved_cell_rows,
            'recently_active_cells': build_stage4_probe_cell_rows(recently_active_cells),
        })
        if state_reset_candidate
        else frozenset(
            coordinate
            for coordinate in unresolved_cells
            if coordinate in carried_post_reset_contaminated_cells
        )
    )

    reason_summary = build_stage4_probe_reason_summary(
        probe_label=probe_label,
        changed_cell_count=changed_cell_count,
        judgeable_cell_count=len(judgeable_cells),
        unresolved_cell_count=unresolved_cell_count,
        judgeable_coverage=judgeable_coverage,
        structural_holding_support=structural_holding_support,
        opening_zone_low_confidence=opening_zone_low_confidence,
        late_resolution_only=late_resolution_only,
    )

    record_probe_timing('classification_seconds', classification_started_at)
    baseline_comparison_updates = {
        coordinate: current_assignments[coordinate]
        for coordinate in confirmed_changed_cells
        if coordinate in current_assignments
    }
    return finalize_probe_evaluation({
        'slice_records': slice_records,
        'footprint': recently_active_cells,
        'footprint_size': len(recently_active_cells),
        'subregions': [],
        'subregion_results': [],
        'summary': None,
        'probe_model': 'cell_level',
        'probe_anchor_time': Timecode(total_frames=slice_end).to_hhmmssff(),
        'probe_local_evaluation_window': {
            'start_frame': slice_start,
            'end_frame': slice_end,
            'start_time': Timecode(total_frames=slice_start).to_hhmmssff(),
            'end_time': Timecode(total_frames=slice_end).to_hhmmssff(),
        },
        'recently_active_cells': build_stage4_probe_cell_rows(recently_active_cells),
        'judgeable_cells': build_stage4_probe_cell_rows(judgeable_cells),
        'contaminated_cells': build_stage4_probe_cell_rows(contaminated_cells),
        'confirmed_changed_cells': confirmed_changed_cell_rows,
        'confirmed_unchanged_cells': confirmed_unchanged_cell_rows,
        'unresolved_cells': unresolved_cell_rows,
        'cell_results': cell_results,
        'post_reset_contaminated_cells': build_stage4_probe_cell_rows(post_reset_contaminated_carry),
        'changed_support_score': changed_support_score,
        'judgeable_changed_support_score': judgeable_changed_support_score,
        'unchanged_support_score': unchanged_support_score,
        'unresolved_support_score': unresolved_support_score,
        'relevant_cell_count': len(recently_active_cells),
        'judgeable_cell_count': len(judgeable_cells),
        'confirmed_changed_cell_count': changed_cell_count,
        'confirmed_unchanged_cell_count': len(confirmed_unchanged_cells),
        'unresolved_cell_count': unresolved_cell_count,
        'judgeable_coverage': judgeable_coverage,
        'structural_holding_support': structural_holding_support,
        'opening_zone_low_confidence': opening_zone_low_confidence,
        'late_resolution_only': late_resolution_only,
        'changed_touch_frame_count': changed_touch_frame_count,
        'changed_touch_frame_ratio': round(changed_touch_frame_ratio, 6),
        'before_window_candidate_count': before_candidate_count,
        'current_window_candidate_count': current_candidate_count,
        'cell_reference_debug': cell_reference_debug,
        'current_window': {
            'search_start': current_search_start,
            'search_end': current_search_end,
            'assigned_cell_count': len(current_assignments),
            'mean_window_activity': round(after_reference_activity, 6),
            'post_union_resolution_end': post_union_resolution_end,
        },
        'classification': classification,
        'reason': reason,
        'reason_summary': reason_summary,
        'lasting_change_evidence_score': changed_support_score,
        'before_reference_activity': 0.0,
        'after_reference_activity': after_reference_activity,
        'reference_windows_reliable': bool(before_assignments) and bool(current_assignments),
        'opening_attribution_start_frame': opening_attribution_start_frame,
        'opening_attribution_start_time': (
            Timecode(total_frames=opening_attribution_start_frame).to_hhmmssff()
            if opening_attribution_start_frame is not None
            else None
        ),
        'baseline_assigned_cell_count': len(baseline_assigned_coordinates),
        'baseline_comparison_update_cell_count': len(baseline_comparison_updates),
        '_baseline_comparison_updates': baseline_comparison_updates,
        'probe_label': probe_label,
        'state_reset_candidate': state_reset_candidate,
    })

def stage4_should_apply_opening_attribution(
    screened_candidate_union: ScreenedCandidateUnion,
    first_probe: ClassifiedTimeSlice,
    first_evaluation: dict[str, object],
) -> bool:
    opening_attribution_start_frame = first_evaluation.get('opening_attribution_start_frame')
    if not isinstance(opening_attribution_start_frame, int):
        return False

    band_start_frame = first_probe.start_frame
    if not (
        screened_candidate_union.candidate_union.start_frame
        <= opening_attribution_start_frame
        < band_start_frame
    ):
        return False

    if bool(first_evaluation.get('opening_zone_low_confidence', False)):
        return True

    return (band_start_frame - opening_attribution_start_frame) <= STAGE4_PROBE_INTERVAL_FRAMES

def build_stage4_band_from_probe_results(
    screened_candidate_union: ScreenedCandidateUnion,
    slice_index: int,
    probe_results: list[tuple[ClassifiedTimeSlice, dict[str, object]]],
) -> ClassifiedTimeSlice:
    first_probe, _ = probe_results[0]
    last_probe, _ = probe_results[-1]
    first_evaluation = probe_results[0][1]
    band_start_frame = first_probe.start_frame
    opening_attribution_start_frame = first_evaluation.get('opening_attribution_start_frame')
    if stage4_should_apply_opening_attribution(screened_candidate_union, first_probe, first_evaluation):
        band_start_frame = opening_attribution_start_frame
    elif (str(first_evaluation.get('probe_label', '')) in ('unclear', 'holding') and bool(first_evaluation.get('opening_zone_low_confidence', False)) and band_start_frame > screened_candidate_union.candidate_union.start_frame):
        band_start_frame = screened_candidate_union.candidate_union.start_frame
    footprint = frozenset(
        coordinate
        for probe_slice, _ in probe_results
        for coordinate in probe_slice.footprint
    )
    within_slice_record_count = sum(probe_slice.within_slice_record_count for probe_slice, _ in probe_results)
    lasting_change_evidence_score = sum(
        float(evaluation.get('changed_support_score', evaluation['lasting_change_evidence_score']))
        for _, evaluation in probe_results
    ) / max(1, len(probe_results))
    before_reference_activity = sum(
        float(evaluation['before_reference_activity'])
        for _, evaluation in probe_results
    ) / max(1, len(probe_results))
    after_reference_activity = sum(
        float(evaluation['after_reference_activity'])
        for _, evaluation in probe_results
    ) / max(1, len(probe_results))
    reference_windows_reliable = all(
        bool(evaluation['reference_windows_reliable'])
        for _, evaluation in probe_results
    )

    return ClassifiedTimeSlice(
        slice_index=slice_index,
        parent_union_index=screened_candidate_union.candidate_union.union_index,
        slice_level=0,
        start_frame=band_start_frame,
        end_frame=last_probe.end_frame,
        start_time=Timecode(total_frames=band_start_frame).to_hhmmssff(),
        end_time=Timecode(total_frames=last_probe.end_frame).to_hhmmssff(),
        footprint=footprint,
        footprint_size=len(footprint),
        within_slice_record_count=within_slice_record_count,
        classification='valid',
        reason='probe_supported_band',
        lasting_change_evidence_score=lasting_change_evidence_score,
        before_reference_activity=before_reference_activity,
        after_reference_activity=after_reference_activity,
        reference_windows_reliable=reference_windows_reliable,
    )

def stage4_probe_supports_band_continuity(evaluation: dict[str, object]) -> bool:
    probe_label = str(evaluation.get('probe_label', 'unclear'))
    if probe_label == 'positive':
        return True
    return bool(evaluation.get('structural_holding_support', False))



def stage4_probe_supports_ambiguous_bridge(
    evaluation: dict[str, object],
    active_band_probes: list[tuple[ClassifiedTimeSlice, dict[str, object]]] | None = None,
) -> bool:
    probe_label = str(evaluation.get('probe_label', 'unclear'))
    if probe_label != 'unclear':
        return False
    if not bool(evaluation.get('structural_holding_support', False)):
        return False

    unresolved_cell_count = int(evaluation.get('unresolved_cell_count', 0))
    if unresolved_cell_count >= 2:
        return True

    if unresolved_cell_count < 1 or not active_band_probes or len(active_band_probes) < 2:
        return False

    last_evaluation = active_band_probes[-1][1]
    return (
        str(last_evaluation.get('probe_label', 'unclear')) == 'positive'
        and bool(last_evaluation.get('structural_holding_support', False))
    )


def stage4_probe_supports_opening_backfill(
    evaluation: dict[str, object],
    anchor_evaluation: dict[str, object] | None = None,
) -> bool:
    probe_label = str(evaluation.get('probe_label', 'unclear'))
    changed_support_score = float(evaluation.get('changed_support_score', 0.0))
    judgeable_changed_support_score = float(evaluation.get('judgeable_changed_support_score', 0.0))
    confirmed_changed_cell_count = int(evaluation.get('confirmed_changed_cell_count', 0))
    unresolved_cell_count = int(evaluation.get('unresolved_cell_count', 0))
    structural_holding_support = bool(evaluation.get('structural_holding_support', False))

    baseline_support = False
    if probe_label == 'holding':
        baseline_support = (
            confirmed_changed_cell_count > 0
            or changed_support_score > 0.0
            or judgeable_changed_support_score > 0.0
            or (structural_holding_support and unresolved_cell_count > 0)
        )
    elif probe_label == 'unclear':
        # Do not let zero-changed ambiguous probes define the front edge of a
        # positive band. They can still bridge once a band is active.
        baseline_support = (
            confirmed_changed_cell_count > 0
            or changed_support_score > 0.0
            or judgeable_changed_support_score > 0.0
        )

    if not baseline_support:
        return False

    if (
        anchor_evaluation is None
        or 'unresolved_cells' not in evaluation
        or 'confirmed_changed_cells' not in anchor_evaluation
    ):
        return baseline_support

    anchor_changed_cells = extract_stage4_probe_coordinates(
        anchor_evaluation.get('confirmed_changed_cells', [])
    )
    unresolved_cells = extract_stage4_probe_coordinates(evaluation.get('unresolved_cells', []))
    if not anchor_changed_cells or not unresolved_cells:
        return baseline_support

    unresolved_has_anchor_connection = stage4_coordinate_sets_have_local_connection(
        unresolved_cells,
        anchor_changed_cells,
    )
    if probe_label == 'holding' and (
        confirmed_changed_cell_count > 0
        or changed_support_score > 0.0
        or judgeable_changed_support_score > 0.0
    ):
        return True

    return unresolved_has_anchor_connection



def extract_stage4_probe_coordinates(cell_rows: Iterable[dict[str, object]]) -> frozenset[GridCoordinate]:
    coordinates: set[GridCoordinate] = set()
    for cell_row in cell_rows:
        coordinate = cell_row.get('coordinate') if isinstance(cell_row, dict) else None
        if (
            isinstance(coordinate, (list, tuple))
            and len(coordinate) == 2
            and all(isinstance(value, int) for value in coordinate)
        ):
            coordinates.add((int(coordinate[0]), int(coordinate[1])))
    return frozenset(coordinates)


def stage4_coordinate_span_is_broad(coordinates: frozenset[GridCoordinate]) -> bool:
    if not coordinates:
        return False
    row_indices = [coordinate[0] for coordinate in coordinates]
    column_indices = [coordinate[1] for coordinate in coordinates]
    row_span = max(row_indices) - min(row_indices)
    column_span = max(column_indices) - min(column_indices)
    return row_span >= STAGE4_STATE_RESET_MIN_ROW_SPAN and column_span >= STAGE4_STATE_RESET_MIN_COLUMN_SPAN



def stage4_probe_is_state_reset_candidate(evaluation: dict[str, object]) -> bool:
    if str(evaluation.get('probe_label', 'unclear')) != 'positive':
        return False

    unresolved_cell_count = int(evaluation.get('unresolved_cell_count', 0))
    changed_cell_count = int(evaluation.get('confirmed_changed_cell_count', 0))
    changed_support_score = float(evaluation.get('changed_support_score', 0.0))
    changed_cells = extract_stage4_probe_coordinates(evaluation.get('confirmed_changed_cells', []))
    if (
        unresolved_cell_count <= 0
        or changed_cell_count < STAGE4_STATE_RESET_MIN_CHANGED_CELLS
        or not changed_cells
    ):
        return False

    changed_clusters = build_stage3_connected_clusters(changed_cells)
    has_reset_shape = (
        len(changed_clusters) >= STAGE4_STATE_RESET_MIN_CHANGED_CLUSTER_COUNT
        or stage4_coordinate_span_is_broad(changed_cells)
    )
    return has_reset_shape and changed_support_score >= STAGE4_STATE_RESET_MIN_CHANGED_SUPPORT



def stage4_extract_post_reset_contaminated_cells(evaluation: dict[str, object]) -> frozenset[GridCoordinate]:
    if not stage4_probe_is_state_reset_candidate(evaluation):
        return frozenset()
    unresolved_cells = extract_stage4_probe_coordinates(evaluation.get('unresolved_cells', []))
    recently_active_cells = extract_stage4_probe_coordinates(evaluation.get('recently_active_cells', []))
    unresolved_neighbors = stage4_collect_locally_connected_coordinates(
        unresolved_cells,
        recently_active_cells.difference(unresolved_cells),
    )
    return frozenset(unresolved_cells.union(unresolved_neighbors))


def stage4_opening_probe_needs_revalidation(
    screened_candidate_union: ScreenedCandidateUnion,
    evaluation: dict[str, object],
    sampled_frames: list[dict[str, object]] | None,
) -> bool:
    if not sampled_frames:
        return False
    if str(evaluation.get('probe_label', 'unclear')) != 'positive':
        return False
    if not bool(evaluation.get('opening_zone_low_confidence', False)):
        return False

    sampled_frame_start = min(int(sample['frame_index']) for sample in sampled_frames)
    if screened_candidate_union.candidate_union.start_frame > (
        sampled_frame_start + STAGE4_OPENING_REVALIDATION_MAX_CHAPTER_OFFSET_FRAMES
    ):
        return False

    changed_cells = extract_stage4_probe_coordinates(evaluation.get('confirmed_changed_cells', []))
    if not changed_cells or len(changed_cells) > STAGE4_OPENING_REVALIDATION_MAX_CHANGED_CELLS:
        return False
    if len(build_stage3_connected_clusters(changed_cells)) != 1:
        return False

    changed_support_score = float(evaluation.get('changed_support_score', 0.0))
    judgeable_changed_support_score = float(evaluation.get('judgeable_changed_support_score', 0.0))
    return (
        changed_support_score <= STAGE4_OPENING_REVALIDATION_MAX_CHANGED_SUPPORT
        and judgeable_changed_support_score <= STAGE4_OPENING_REVALIDATION_MAX_CHANGED_SUPPORT
    )


def stage4_opening_probe_collapses_under_later_revalidation(
    target_changed_cells: frozenset[GridCoordinate],
    probe_results: list[tuple[ClassifiedTimeSlice, dict[str, object]]],
    probe_index: int,
) -> bool:
    if not target_changed_cells:
        return False

    lookahead_results = probe_results[
        probe_index + 1 : probe_index + 1 + STAGE4_OPENING_REVALIDATION_LOOKAHEAD_PROBES
    ]
    for _, later_evaluation in lookahead_results:
        later_cell_results = {
            tuple(cell_result['coordinate']): str(cell_result.get('classification', 'unresolved'))
            for cell_result in later_evaluation.get('cell_results', [])
            if isinstance(cell_result, dict)
            and isinstance(cell_result.get('coordinate'), list)
            and len(cell_result['coordinate']) == 2
        }
        if any(
            later_cell_results.get(coordinate) == 'confirmed_changed'
            for coordinate in target_changed_cells
        ):
            return False

        unchanged_matches = sum(
            1
            for coordinate in target_changed_cells
            if later_cell_results.get(coordinate) == 'confirmed_unchanged'
        )
        if unchanged_matches <= 0:
            continue

        unchanged_match_ratio = unchanged_matches / max(1, len(target_changed_cells))
        if unchanged_match_ratio >= STAGE4_OPENING_REVALIDATION_MIN_UNCHANGED_MATCH_RATIO:
            return True
    return False


def stage4_apply_opening_probe_revalidation(
    screened_candidate_union: ScreenedCandidateUnion,
    probe_results: list[tuple[ClassifiedTimeSlice, dict[str, object]]],
    sampled_frames: list[dict[str, object]] | None,
) -> list[tuple[ClassifiedTimeSlice, dict[str, object]]]:
    rewritten_probe_results = list(probe_results)
    for probe_index, (probe_slice, evaluation) in enumerate(rewritten_probe_results):
        if not stage4_opening_probe_needs_revalidation(
            screened_candidate_union,
            evaluation,
            sampled_frames,
        ):
            continue

        target_changed_cells = extract_stage4_probe_coordinates(evaluation.get('confirmed_changed_cells', []))
        if not stage4_opening_probe_collapses_under_later_revalidation(
            target_changed_cells,
            rewritten_probe_results,
            probe_index,
        ):
            break

        rewritten_evaluation = dict(evaluation)
        rewritten_evaluation['classification'] = 'invalid'
        rewritten_evaluation['reason'] = 'probe_cell_opening_revalidation_failed'
        rewritten_evaluation['reason_summary'] = ('negative: opening low-confidence changed cells collapsed under later clean revalidation;')
        rewritten_evaluation['probe_label'] = 'negative'
        rewritten_evaluation['lasting_change_evidence_score'] = 0.0
        rewritten_evaluation['opening_revalidation_rejected'] = True
        rewritten_probe = replace(
            probe_slice,
            classification='invalid',
            reason='probe_cell_opening_revalidation_failed',
            lasting_change_evidence_score=0.0,
        )
        rewritten_probe_results[probe_index] = (rewritten_probe, rewritten_evaluation)
        break
    return rewritten_probe_results

def build_stage4_bands_from_probe_results(
    screened_candidate_union: ScreenedCandidateUnion,
    probe_results: list[tuple[ClassifiedTimeSlice, dict[str, object]]],
) -> list[ClassifiedTimeSlice]:
    bands: list[ClassifiedTimeSlice] = []
    active_band_probes: list[tuple[ClassifiedTimeSlice, dict[str, object]]] = []
    pending_holding_probes: list[tuple[ClassifiedTimeSlice, dict[str, object]]] = []

    def close_active_band() -> None:
        if not active_band_probes:
            return
        bands.append(
            build_stage4_band_from_probe_results(
                screened_candidate_union,
                len(bands) + 1,
                active_band_probes,
            )
        )
        active_band_probes.clear()

    for probe_slice, evaluation in probe_results:
        probe_label = str(evaluation['probe_label'])
        continuity_support = stage4_probe_supports_band_continuity(evaluation)
        late_resolution_only = bool(evaluation.get('late_resolution_only', False))

        if late_resolution_only:
            if not active_band_probes and pending_holding_probes:
                prepend_start = len(pending_holding_probes)
                while prepend_start > 0 and stage4_probe_supports_opening_backfill(
                    pending_holding_probes[prepend_start - 1][1],
                    evaluation,
                ):
                    prepend_start -= 1
                if prepend_start < len(pending_holding_probes):
                    active_band_probes.extend(pending_holding_probes[prepend_start:])
                    close_active_band()
            elif active_band_probes:
                close_active_band()
            pending_holding_probes.clear()
            continue

        if probe_label == 'positive':
            if pending_holding_probes and not active_band_probes:
                prepend_start = len(pending_holding_probes)
                while prepend_start > 0 and stage4_probe_supports_opening_backfill(
                    pending_holding_probes[prepend_start - 1][1],
                    evaluation,
                ):
                    prepend_start -= 1
                if prepend_start < len(pending_holding_probes):
                    active_band_probes.extend(pending_holding_probes[prepend_start:])
            pending_holding_probes.clear()
            active_band_probes.append((probe_slice, evaluation))
            continue

        if probe_label in ('holding', 'unclear'):
            if continuity_support:
                if active_band_probes:
                    if probe_label == 'holding' or stage4_probe_supports_ambiguous_bridge(evaluation, active_band_probes):
                        active_band_probes.append((probe_slice, evaluation))
                    else:
                        close_active_band()
                        pending_holding_probes.clear()
                else:
                    pending_holding_probes.append((probe_slice, evaluation))
                continue
            close_active_band()
            pending_holding_probes.clear()
            continue

        close_active_band()
        pending_holding_probes.clear()

    close_active_band()
    return bands

def classify_stage4_time_slices_with_subregion_debug(
    screened_candidate_unions: Iterable[ScreenedCandidateUnion],
    records: Iterable[MovementEvidenceRecord],
    sampled_frames: list[dict[str, object]] | None = None,
    settings: DetectorSettings | None = None,
) -> tuple[list[ClassifiedTimeSlice], list[dict[str, object]], list[dict[str, object]]]:
    ordered_records = sorted(records, key=lambda record: record.frame_index)
    classified_slices: list[ClassifiedTimeSlice] = []
    stage4_subregion_debug: list[dict[str, object]] = []
    stage4_cell_reference_debug: list[dict[str, object]] = []

    for screened_candidate_union in screened_candidate_unions:
        if not screened_candidate_union.surviving:
            continue

        slice_ranges = build_stage4_time_slice_ranges(screened_candidate_union)
        probe_results: list[tuple[ClassifiedTimeSlice, dict[str, object]]] = []
        carried_unresolved_cells: frozenset[GridCoordinate] = frozenset()
        carried_recently_changed_cells: frozenset[GridCoordinate] = frozenset()
        carried_post_reset_contaminated_cells: frozenset[GridCoordinate] = frozenset()
        previous_probe_blocks_opening_attribution = False
        baseline_comparison_state: dict[GridCoordinate, dict[str, object]] = {}
        previous_probe_end: int | None = None
        union_baseline_assignment_seconds = 0.0

        if sampled_frames and settings is not None and screened_candidate_union.candidate_union.union_footprint:
            try:
                import cv2  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError('OpenCV is required for staged cell-level probe classification.') from exc

            union_baseline_assignment_started_at = time.perf_counter()
            sampled_frame_start = min(int(sample['frame_index']) for sample in sampled_frames)
            reference_canvas_shape = get_stage3_canvas_shape(sampled_frames[0])
            union_footprint = screened_candidate_union.candidate_union.union_footprint
            union_baseline_assignments, _ = select_stage3_cell_reference_windows(
                search_start=max(sampled_frame_start, screened_candidate_union.candidate_union.start_frame - STAGE3_ART_STATE_RESCUE_SEARCH_FRAMES),
                search_end=max(sampled_frame_start, screened_candidate_union.candidate_union.start_frame),
                prefer_nearest=True,
                footprint=union_footprint,
                endpoint_coordinates=frozenset(),
                require_external_movement_for_all_cells=False,
                require_external_movement_for_endpoints=False,
                require_external_movement_outside_coordinate_for_all_cells=False,
                ordered_records=ordered_records,
                sampled_frames=sampled_frames,
                cell_art_state_masks=build_stage3_cell_art_state_masks(union_footprint, reference_canvas_shape),
                settings=settings,
                cv2=cv2,
                stage3_context=build_stage3_runtime_context(ordered_records, sampled_frames),
                target_coordinates=union_footprint,
            )
            baseline_comparison_state.update(union_baseline_assignments)
            union_baseline_assignment_seconds = round(time.perf_counter() - union_baseline_assignment_started_at, 6)

        for probe_index, (slice_start, slice_end) in enumerate(slice_ranges, start=1):
            evaluation = evaluate_stage4_cell_level_probe(
                screened_candidate_union=screened_candidate_union,
                slice_start=slice_start,
                slice_end=slice_end,
                ordered_records=ordered_records,
                sampled_frames=sampled_frames,
                settings=settings,
                carried_unresolved_cells=carried_unresolved_cells,
                carried_recently_changed_cells=carried_recently_changed_cells,
                carried_post_reset_contaminated_cells=carried_post_reset_contaminated_cells,
                baseline_comparison_state=baseline_comparison_state,
                previous_probe_end=previous_probe_end,
                previous_probe_blocks_opening_attribution=previous_probe_blocks_opening_attribution,
            )
            probe_slice = build_classified_time_slice_from_evaluation(
                screened_candidate_union=screened_candidate_union,
                slice_index=probe_index,
                slice_level=0,
                slice_start=slice_start,
                slice_end=slice_end,
                evaluation=evaluation,
            )
            evaluation['stage4_union_baseline_assignment_seconds'] = union_baseline_assignment_seconds
            debug_entry = serialize_stage4_time_slice_subregion_debug_entry(probe_slice, evaluation)
            stage4_subregion_debug.append(debug_entry)
            stage4_cell_reference_debug.extend(
                serialize_stage4_cell_reference_debug_entries(probe_slice, evaluation)
            )
            probe_results.append((probe_slice, evaluation))

            baseline_comparison_state.update(evaluation.get('_baseline_comparison_updates', {}))
            carried_recently_changed_cells = frozenset(
                tuple(cell_row['coordinate'])
                for cell_row in evaluation.get('confirmed_changed_cells', [])
                if isinstance(cell_row, dict) and 'coordinate' in cell_row
            )
            carried_unresolved_cells = frozenset(
                tuple(cell_row['coordinate'])
                for cell_row in evaluation.get('unresolved_cells', [])
                if isinstance(cell_row, dict) and 'coordinate' in cell_row
            )
            carried_post_reset_contaminated_cells = frozenset(
                tuple(cell_row['coordinate'])
                for cell_row in evaluation.get('post_reset_contaminated_cells', [])
                if isinstance(cell_row, dict) and 'coordinate' in cell_row
            )
            if str(evaluation.get('probe_label')) == 'negative':
                carried_unresolved_cells = frozenset()
                carried_recently_changed_cells = frozenset()
                carried_post_reset_contaminated_cells = frozenset()
            previous_probe_blocks_opening_attribution = (
                str(evaluation.get('probe_label')) == 'negative'
                and str(evaluation.get('reason')) == 'probe_cell_broad_unchanged_support'
            )
            previous_probe_end = slice_end

        probe_results = stage4_apply_opening_probe_revalidation(
            screened_candidate_union,
            probe_results,
            sampled_frames,
        )
        classified_slices.extend(
            build_stage4_bands_from_probe_results(screened_candidate_union, probe_results)
        )

    return classified_slices, stage4_subregion_debug, stage4_cell_reference_debug

def classify_stage4_time_slices(
    screened_candidate_unions: Iterable[ScreenedCandidateUnion],
    records: Iterable[MovementEvidenceRecord],
    sampled_frames: list[dict[str, object]] | None = None,
    settings: DetectorSettings | None = None,
) -> list[ClassifiedTimeSlice]:
    classified_slices, _, _ = classify_stage4_time_slices_with_subregion_debug(
        screened_candidate_unions,
        records,
        sampled_frames=sampled_frames,
        settings=settings,
    )
    return classified_slices


# ============================================================
# SECTION H - Stage 5 Recursive Sub-Slice Refinement
# ============================================================


def build_stage5_sub_slice_ranges(time_slice: ClassifiedTimeSlice) -> list[tuple[int, int]]:
    slice_start = time_slice.start_frame
    slice_end = time_slice.end_frame
    if slice_end <= slice_start:
        return []

    midpoint = slice_start + ((slice_end - slice_start) // 2)
    if midpoint <= slice_start or midpoint >= slice_end:
        return [(slice_start, slice_end)]
    return [(slice_start, midpoint), (midpoint, slice_end)]



def classify_stage5_minimum_size_leaf(
    time_slice: ClassifiedTimeSlice,
    allow_active_reference_rescue: bool = False,
    allow_long_strong_union_rocky_rescue: bool = False,
    allow_high_parent_activity_rescue: bool = False,
    allow_reference_unreliable_rescue: bool = False,
    allow_structural_gap_rescue: bool = False,
    allow_art_state_supported_rescue: bool = False,
    allow_valid_anchor_promotion: bool = False,
    has_adjacent_valid_support: bool = False,
) -> ClassifiedTimeSlice:
    if time_slice.classification in {'valid', 'invalid', 'boundary'}:
        return time_slice
    if time_slice.footprint_size <= 0:
        return replace(time_slice, classification='invalid', reason='minimum_subdivision_size_reached')
    if has_adjacent_valid_support:
        return replace(time_slice, classification='boundary', reason='minimum_subdivision_size_reached')
    return replace(time_slice, classification='invalid', reason='minimum_subdivision_size_reached')


def build_stage5_ordered_candidate_cells(
    time_slice: ClassifiedTimeSlice,
    ordered_records: list[MovementEvidenceRecord],
    *,
    search_start: int,
    search_end: int,
    prefer_latest: bool,
    max_candidate_cells: int = STAGE5_MAX_CANDIDATE_CELLS,
) -> list[tuple[GridCoordinate, int]]:
    candidate_frames: dict[GridCoordinate, int] = {}
    for record in collect_records_in_frame_range(ordered_records, search_start, search_end):
        if not record.movement_present:
            continue
        for coordinate in record.touched_grid_coordinates:
            if coordinate not in time_slice.footprint:
                continue
            frame_index = int(record.frame_index)
            if prefer_latest:
                candidate_frames[coordinate] = max(frame_index, candidate_frames.get(coordinate, frame_index))
            else:
                candidate_frames[coordinate] = min(frame_index, candidate_frames.get(coordinate, frame_index))

    ordered_candidates = sorted(
        candidate_frames.items(),
        key=lambda item: (
            -item[1] if prefer_latest else item[1],
            item[0][0],
            item[0][1],
        ),
    )
    return ordered_candidates[:max_candidate_cells]


def build_stage5_candidate_touch_frames(
    time_slice: ClassifiedTimeSlice,
    ordered_records: list[MovementEvidenceRecord],
    *,
    search_start: int,
    search_end: int,
) -> dict[GridCoordinate, list[int]]:
    candidate_frames: dict[GridCoordinate, list[int]] = {}
    for record in collect_records_in_frame_range(ordered_records, search_start, search_end):
        if not record.movement_present:
            continue
        for coordinate in record.touched_grid_coordinates:
            if coordinate not in time_slice.footprint:
                continue
            candidate_frames.setdefault(coordinate, []).append(int(record.frame_index))
    return candidate_frames


def find_stage5_broad_movement_boundary_frame(
    time_slice: ClassifiedTimeSlice,
    ordered_records: list[MovementEvidenceRecord],
    *,
    search_start: int,
    search_end: int,
    prefer_latest: bool,
) -> int | None:
    minimum_touch_count = max(
        STAGE5_BROAD_MOVEMENT_MIN_CELLS,
        int(np.ceil(time_slice.footprint_size * STAGE5_BROAD_MOVEMENT_MIN_FOOTPRINT_RATIO)),
    )
    qualifying_frames: list[int] = []
    for record in collect_records_in_frame_range(ordered_records, search_start, search_end):
        if not record.movement_present:
            continue
        touched_coordinates = frozenset(
            coordinate
            for coordinate in record.touched_grid_coordinates
            if coordinate in time_slice.footprint
        )
        if len(touched_coordinates) < minimum_touch_count:
            continue
        if not stage4_coordinate_span_is_broad(touched_coordinates):
            continue
        qualifying_frames.append(int(record.frame_index))
    if not qualifying_frames:
        return None
    qualifying_runs: list[tuple[int, int]] = []
    run_start = qualifying_frames[0]
    run_end = qualifying_frames[0]
    for frame_index in qualifying_frames[1:]:
        if frame_index <= (run_end + 1):
            run_end = frame_index
            continue
        qualifying_runs.append((run_start, run_end))
        run_start = frame_index
        run_end = frame_index
    qualifying_runs.append((run_start, run_end))

    eligible_runs = [
        (run_start, run_end)
        for run_start, run_end in qualifying_runs
        if ((run_end - run_start) + 1) >= STAGE5_BROAD_MOVEMENT_MIN_CONSECUTIVE_FRAMES
    ]
    if not eligible_runs:
        return None
    if prefer_latest:
        return eligible_runs[-1][1]
    return eligible_runs[0][0]


def stage5_trim_respects_minimum_clip_length(
    *,
    start_frame: int,
    end_frame: int,
    settings: DetectorSettings,
) -> bool:
    return (end_frame - start_frame) >= settings.min_clip_length.total_frames


def build_stage5_trimmed_time_slice(
    time_slice: ClassifiedTimeSlice,
    *,
    start_frame: int,
    end_frame: int,
    ordered_records: list[MovementEvidenceRecord],
    stage5_debug: dict[str, object] | None = None,
) -> ClassifiedTimeSlice:
    return replace(
        time_slice,
        start_frame=start_frame,
        end_frame=end_frame,
        start_time=Timecode(total_frames=start_frame).to_hhmmssff(),
        end_time=Timecode(total_frames=end_frame).to_hhmmssff(),
        within_slice_record_count=len(collect_records_in_frame_range(ordered_records, start_frame, end_frame)),
        stage5_debug=stage5_debug,
    )


def find_stage5_valid_slice_start_trim_frame(
    time_slice: ClassifiedTimeSlice,
    screened_candidate_union: ScreenedCandidateUnion,
    ordered_records: list[MovementEvidenceRecord],
    sampled_frames: list[dict[str, object]],
    settings: DetectorSettings,
    cv2,
) -> tuple[int | None, list[dict[str, object]]]:
    start_search_end = min(time_slice.end_frame, time_slice.start_frame + STAGE5_BOUNDARY_SEARCH_FRAMES)
    trim_limit_end = min(time_slice.end_frame, time_slice.start_frame + STAGE5_MAX_TRIM_FRAMES)
    if (start_search_end - time_slice.start_frame) < STAGE5_CONFIRMATION_WINDOW_FRAMES:
        return None, []

    ordered_candidates = build_stage5_ordered_candidate_cells(
        time_slice,
        ordered_records,
        search_start=time_slice.start_frame,
        search_end=start_search_end,
        prefer_latest=False,
    )
    touch_frames_by_coordinate = build_stage5_candidate_touch_frames(
        time_slice,
        ordered_records,
        search_start=time_slice.start_frame,
        search_end=start_search_end,
    )
    if not ordered_candidates:
        return None, []

    sampled_frame_start = min(int(sample['frame_index']) for sample in sampled_frames)
    reference_canvas_shape = get_stage3_canvas_shape(sampled_frames[0])
    cell_art_state_masks = build_stage3_cell_art_state_masks(time_slice.footprint, reference_canvas_shape)
    stage3_context = build_stage3_runtime_context(ordered_records, sampled_frames)
    baseline_cache: dict[tuple[int, int], tuple[list[dict[str, object]], object | None]] = {}
    comparison_cache: dict[tuple[int, int, int, int], dict[str, object] | None] = {}
    before_assignments, _ = select_stage3_cell_reference_windows(
        search_start=max(sampled_frame_start, screened_candidate_union.candidate_union.start_frame - STAGE3_ART_STATE_RESCUE_SEARCH_FRAMES),
        search_end=max(sampled_frame_start, time_slice.start_frame),
        prefer_nearest=True,
        footprint=time_slice.footprint,
        endpoint_coordinates=frozenset(),
        require_external_movement_for_all_cells=False,
        require_external_movement_for_endpoints=False,
        require_external_movement_outside_coordinate_for_all_cells=False,
        ordered_records=ordered_records,
        sampled_frames=sampled_frames,
        cell_art_state_masks=cell_art_state_masks,
        settings=settings,
        cv2=cv2,
        stage3_context=stage3_context,
        target_coordinates=frozenset(coordinate for coordinate, _ in ordered_candidates),
    )
    if not before_assignments:
        return None, []

    earliest_positive_frame: int | None = None
    latest_clean_frame_for_earliest_positive: int | None = None
    candidate_outcomes: list[dict[str, object]] = []
    for coordinate, first_touch_frame in ordered_candidates:
        before_window = before_assignments.get(coordinate)
        touch_frames = touch_frames_by_coordinate.get(coordinate, [])
        outcome: dict[str, object] = {
            'label': format_grid_coordinate_label(coordinate),
            'coordinate': list(coordinate),
            'first_touch_frame': first_touch_frame,
            'first_touch_time': Timecode(total_frames=first_touch_frame).to_hhmmssff(),
            'attempted_touch_count': len(touch_frames),
        }
        if before_window is None:
            outcome['result'] = 'no_before_reference'
            candidate_outcomes.append(outcome)
            continue
        positive_frame: int | None = None
        positive_score: float | None = None
        for touch_frame in touch_frames:
            for current_window_start in range(
                max(time_slice.start_frame, touch_frame),
                max(time_slice.start_frame, start_search_end - STAGE5_CONFIRMATION_WINDOW_FRAMES) + 1,
            ):
                current_window_end = min(start_search_end, current_window_start + STAGE5_CONFIRMATION_WINDOW_FRAMES)
                if (current_window_end - current_window_start) < 2:
                    continue
                current_window = build_stage3_reference_window_metadata(
                    current_window_start,
                    current_window_end,
                    ordered_records,
                    sampled_frames,
                    stage3_context,
                )
                change_score = compute_stage4_cell_change_score(
                    coordinate,
                    before_window,
                    current_window,
                    sampled_frames,
                    cell_art_state_masks,
                    baseline_cache,
                    comparison_cache,
                    settings,
                    cv2,
                    stage3_context,
                )
                if change_score is None or change_score < 0.50:
                    continue
                positive_frame = current_window_start
                positive_score = float(change_score)
                break
            if positive_frame is not None:
                break
        if positive_frame is None:
            outcome['result'] = 'no_positive_touch_found'
            candidate_outcomes.append(outcome)
            continue

        outcome['positive_frame'] = positive_frame
        outcome['positive_time'] = Timecode(total_frames=positive_frame).to_hhmmssff()
        outcome['positive_score'] = round(float(positive_score), 6) if positive_score is not None else None

        proposed_trim_frame = min(trim_limit_end, positive_frame)
        clean_frame: int | None = None
        latest_clean_frame = min(proposed_trim_frame, positive_frame - 1)
        for candidate_last_clean_frame in range(latest_clean_frame, time_slice.start_frame - 1, -1):
            candidate_window_end = candidate_last_clean_frame + 1
            earliest_window_start = max(
                time_slice.start_frame,
                candidate_window_end - STAGE5_CONFIRMATION_WINDOW_FRAMES,
            )
            for candidate_window_start in range(earliest_window_start, candidate_window_end - STAGE3_ART_STATE_MIN_SAMPLES + 1):
                if (candidate_window_end - candidate_window_start) < STAGE3_ART_STATE_MIN_SAMPLES:
                    continue
                if not stage3_cell_is_trustworthy_in_window(
                    coordinate,
                    candidate_window_start,
                    candidate_window_end,
                    ordered_records,
                    sampled_frames,
                    cell_art_state_masks,
                    time_slice.footprint,
                    False,
                    settings,
                    cv2,
                    stage3_context,
                ):
                    continue
                clean_frame = candidate_last_clean_frame
                break
            if clean_frame is not None:
                break
        if clean_frame is None:
            outcome['result'] = 'no_clean_frame_found'
            candidate_outcomes.append(outcome)
            continue

        outcome['proposed_trim_frame'] = clean_frame
        outcome['proposed_trim_time'] = Timecode(total_frames=clean_frame).to_hhmmssff()

        if (
            earliest_positive_frame is None
            or positive_frame < earliest_positive_frame
            or (
                positive_frame == earliest_positive_frame
                and (
                    latest_clean_frame_for_earliest_positive is None
                    or clean_frame > latest_clean_frame_for_earliest_positive
                )
            )
        ):
            earliest_positive_frame = positive_frame
            latest_clean_frame_for_earliest_positive = clean_frame
        outcome['result'] = 'qualified'
        candidate_outcomes.append(outcome)
    if latest_clean_frame_for_earliest_positive is None:
        return None, candidate_outcomes
    return latest_clean_frame_for_earliest_positive, candidate_outcomes


def find_stage5_valid_slice_end_trim_frame(
    time_slice: ClassifiedTimeSlice,
    screened_candidate_union: ScreenedCandidateUnion,
    ordered_records: list[MovementEvidenceRecord],
    sampled_frames: list[dict[str, object]],
    settings: DetectorSettings,
    cv2,
) -> tuple[int | None, list[dict[str, object]]]:
    end_search_start = max(time_slice.start_frame, time_slice.end_frame - STAGE5_BOUNDARY_SEARCH_FRAMES)
    trim_limit_start = max(time_slice.start_frame, time_slice.end_frame - STAGE5_MAX_TRIM_FRAMES)
    if (time_slice.end_frame - end_search_start) < STAGE5_CONFIRMATION_WINDOW_FRAMES:
        return None, []

    ordered_candidates = build_stage5_ordered_candidate_cells(
        time_slice,
        ordered_records,
        search_start=end_search_start,
        search_end=time_slice.end_frame,
        prefer_latest=True,
    )
    touch_frames_by_coordinate = build_stage5_candidate_touch_frames(
        time_slice,
        ordered_records,
        search_start=end_search_start,
        search_end=time_slice.end_frame,
    )
    if not ordered_candidates:
        return None, []

    sampled_frame_start = min(int(sample['frame_index']) for sample in sampled_frames)
    reference_canvas_shape = get_stage3_canvas_shape(sampled_frames[0])
    cell_art_state_masks = build_stage3_cell_art_state_masks(time_slice.footprint, reference_canvas_shape)
    stage3_context = build_stage3_runtime_context(ordered_records, sampled_frames)
    baseline_cache: dict[tuple[int, int], tuple[list[dict[str, object]], object | None]] = {}
    comparison_cache: dict[tuple[int, int, int, int], dict[str, object] | None] = {}
    before_assignments, _ = select_stage3_cell_reference_windows(
        search_start=max(sampled_frame_start, screened_candidate_union.candidate_union.start_frame - STAGE3_ART_STATE_RESCUE_SEARCH_FRAMES),
        search_end=max(sampled_frame_start, time_slice.start_frame),
        prefer_nearest=True,
        footprint=time_slice.footprint,
        endpoint_coordinates=frozenset(),
        require_external_movement_for_all_cells=False,
        require_external_movement_for_endpoints=False,
        require_external_movement_outside_coordinate_for_all_cells=False,
        ordered_records=ordered_records,
        sampled_frames=sampled_frames,
        cell_art_state_masks=cell_art_state_masks,
        settings=settings,
        cv2=cv2,
        stage3_context=stage3_context,
        target_coordinates=frozenset(coordinate for coordinate, _ in ordered_candidates),
    )
    if not before_assignments:
        return None, []

    latest_proven_end_frame: int | None = None
    candidate_outcomes: list[dict[str, object]] = []
    for coordinate, last_touch_frame in ordered_candidates:
        before_window = before_assignments.get(coordinate)
        touch_frames = list(reversed(touch_frames_by_coordinate.get(coordinate, [])))
        outcome: dict[str, object] = {
            'label': format_grid_coordinate_label(coordinate),
            'coordinate': list(coordinate),
            'last_touch_frame': last_touch_frame,
            'last_touch_time': Timecode(total_frames=last_touch_frame).to_hhmmssff(),
            'attempted_touch_count': len(touch_frames),
        }
        if before_window is None:
            outcome['result'] = 'no_before_reference'
            candidate_outcomes.append(outcome)
            continue
        candidate_end_frame: int | None = None
        candidate_score: float | None = None
        upper_bound_end_frame = min(
            time_slice.end_frame,
            (max(touch_frames) + 1) if touch_frames else (last_touch_frame + 1),
        )
        for proposed_end_frame in range(upper_bound_end_frame, trim_limit_start - 1, -1):
            current_window_start = max(time_slice.start_frame, proposed_end_frame - STAGE5_CONFIRMATION_WINDOW_FRAMES)
            current_window_end = proposed_end_frame
            if (current_window_end - current_window_start) < 2:
                continue
            current_window = build_stage3_reference_window_metadata(
                current_window_start,
                current_window_end,
                ordered_records,
                sampled_frames,
                stage3_context,
            )
            change_score = compute_stage4_cell_change_score(
                coordinate,
                before_window,
                current_window,
                sampled_frames,
                cell_art_state_masks,
                baseline_cache,
                comparison_cache,
                settings,
                cv2,
                stage3_context,
            )
            if change_score is None or change_score < 0.50:
                continue
            candidate_end_frame = proposed_end_frame
            candidate_score = float(change_score)
            break
        if candidate_end_frame is None:
            outcome['result'] = 'no_positive_touch_found'
            candidate_outcomes.append(outcome)
            continue
        outcome['proposed_trim_frame'] = candidate_end_frame
        outcome['proposed_trim_time'] = Timecode(total_frames=candidate_end_frame).to_hhmmssff()
        outcome['positive_score'] = round(float(candidate_score), 6) if candidate_score is not None else None
        if latest_proven_end_frame is None or candidate_end_frame > latest_proven_end_frame:
            latest_proven_end_frame = candidate_end_frame
        outcome['result'] = 'qualified'
        candidate_outcomes.append(outcome)
    if latest_proven_end_frame is None:
        return None, candidate_outcomes
    return latest_proven_end_frame, candidate_outcomes


def stage5_slices_are_directly_adjacent(first_slice: ClassifiedTimeSlice, second_slice: ClassifiedTimeSlice) -> bool:
    return first_slice.end_frame == second_slice.start_frame or second_slice.end_frame == first_slice.start_frame



def build_stage5_terminal_frontier_groups(
    terminal_slices: Iterable[ClassifiedTimeSlice],
) -> list[list[ClassifiedTimeSlice]]:
    frontier_buckets: dict[tuple[int, int, tuple[int, int] | None], list[ClassifiedTimeSlice]] = {}
    for terminal_slice in terminal_slices:
        frontier_key = (
            terminal_slice.parent_union_index,
            terminal_slice.slice_level,
            terminal_slice.parent_range,
        )
        frontier_buckets.setdefault(frontier_key, []).append(terminal_slice)

    frontier_groups: list[list[ClassifiedTimeSlice]] = []
    for bucket_slices in frontier_buckets.values():
        ordered_bucket_slices = sorted(bucket_slices, key=lambda slice_info: (slice_info.start_frame, slice_info.end_frame))
        current_group: list[ClassifiedTimeSlice] = []
        for bucket_slice in ordered_bucket_slices:
            if not current_group:
                current_group = [bucket_slice]
                continue
            if stage5_slices_are_directly_adjacent(current_group[-1], bucket_slice):
                current_group.append(bucket_slice)
                continue
            frontier_groups.append(current_group)
            current_group = [bucket_slice]
        if current_group:
            frontier_groups.append(current_group)

    return frontier_groups



def resolve_stage5_terminal_slices(
    refined_slices: list[ClassifiedTimeSlice],
) -> list[ClassifiedTimeSlice]:
    valid_slices_by_union: dict[int, list[ClassifiedTimeSlice]] = {}
    for refined_slice in refined_slices:
        if refined_slice.classification == 'valid':
            valid_slices_by_union.setdefault(refined_slice.parent_union_index, []).append(refined_slice)

    terminal_undetermined_slices = [
        refined_slice
        for refined_slice in refined_slices
        if refined_slice.classification == 'undetermined'
    ]
    resolved_lookup: dict[tuple[int, int, int, int], ClassifiedTimeSlice] = {}
    for frontier_group in build_stage5_terminal_frontier_groups(terminal_undetermined_slices):
        if not frontier_group:
            continue
        parent_union_index = frontier_group[0].parent_union_index
        valid_slices = valid_slices_by_union.get(parent_union_index, [])
        for frontier_slice in frontier_group:
            adjacent_valid_support = any(
                stage5_slices_are_directly_adjacent(frontier_slice, valid_slice)
                for valid_slice in valid_slices
            )
            resolved_slice = classify_stage5_minimum_size_leaf(
                frontier_slice,
                has_adjacent_valid_support=adjacent_valid_support,
            )
            resolved_lookup[(
                resolved_slice.parent_union_index,
                resolved_slice.slice_level,
                resolved_slice.start_frame,
                resolved_slice.end_frame,
            )] = resolved_slice

    resolved_slices: list[ClassifiedTimeSlice] = []
    for refined_slice in refined_slices:
        slice_key = (
            refined_slice.parent_union_index,
            refined_slice.slice_level,
            refined_slice.start_frame,
            refined_slice.end_frame,
        )
        resolved_slices.append(resolved_lookup.get(slice_key, refined_slice))
    return resolved_slices
def refine_stage5_sub_slices(
    screened_candidate_unions: Iterable[ScreenedCandidateUnion],
    classified_time_slices: Iterable[ClassifiedTimeSlice],
    records: Iterable[MovementEvidenceRecord],
    sampled_frames: list[dict[str, object]] | None = None,
    settings: DetectorSettings | None = None,
    minimum_subdivision_frames: int = STAGE5_MIN_SUBDIVISION_FRAMES,
) -> list[ClassifiedTimeSlice]:
    ordered_slices = sorted(
        classified_time_slices,
        key=lambda time_slice: (
            time_slice.start_frame,
            time_slice.end_frame,
            time_slice.parent_union_index,
            time_slice.slice_index,
        ),
    )
    if not sampled_frames or settings is None:
        return ordered_slices

    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('OpenCV is required for Stage 5 boundary refinement.') from exc

    ordered_records = sorted(records, key=lambda record: record.frame_index)
    screened_union_lookup = {
        screened_union.candidate_union.union_index: screened_union
        for screened_union in screened_candidate_unions
    }
    refined_slices: list[ClassifiedTimeSlice] = []
    for time_slice in ordered_slices:
        if time_slice.classification != 'valid':
            refined_slices.append(time_slice)
            continue

        screened_candidate_union = screened_union_lookup.get(time_slice.parent_union_index)
        if screened_candidate_union is None or not time_slice.footprint:
            refined_slices.append(time_slice)
            continue

        start_search_end = min(time_slice.end_frame, time_slice.start_frame + STAGE5_BOUNDARY_SEARCH_FRAMES)
        end_search_start = max(time_slice.start_frame, time_slice.end_frame - STAGE5_BOUNDARY_SEARCH_FRAMES)
        stage5_debug: dict[str, object] = {
            'original_start_frame': time_slice.start_frame,
            'original_end_frame': time_slice.end_frame,
            'original_start_time': time_slice.start_time,
            'original_end_time': time_slice.end_time,
            'start_search_start_frame': time_slice.start_frame,
            'start_search_end_frame': start_search_end,
            'end_search_start_frame': end_search_start,
            'end_search_end_frame': time_slice.end_frame,
            'start_candidate_cells': [],
            'end_candidate_cells': [],
            'start_candidate_outcomes': [],
            'end_candidate_outcomes': [],
            'start_broad_movement_frame': None,
            'end_broad_movement_frame': None,
            'start_local_candidate_frame': None,
            'end_local_candidate_frame': None,
            'start_trim_applied': False,
            'end_trim_applied': False,
            'start_trim_reason': 'no_trim',
            'end_trim_reason': 'no_trim',
            'start_winning_path': 'none',
            'end_winning_path': 'none',
        }
        start_candidates = build_stage5_ordered_candidate_cells(
            time_slice,
            ordered_records,
            search_start=time_slice.start_frame,
            search_end=start_search_end,
            prefer_latest=False,
        )
        end_candidates = build_stage5_ordered_candidate_cells(
            time_slice,
            ordered_records,
            search_start=end_search_start,
            search_end=time_slice.end_frame,
            prefer_latest=True,
        )
        stage5_debug['start_candidate_cells'] = [
            {
                'label': format_grid_coordinate_label(coordinate),
                'coordinate': list(coordinate),
                'candidate_frame': frame_index,
                'candidate_time': Timecode(total_frames=frame_index).to_hhmmssff(),
            }
            for coordinate, frame_index in start_candidates
        ]
        stage5_debug['end_candidate_cells'] = [
            {
                'label': format_grid_coordinate_label(coordinate),
                'coordinate': list(coordinate),
                'candidate_frame': frame_index,
                'candidate_time': Timecode(total_frames=frame_index).to_hhmmssff(),
            }
            for coordinate, frame_index in end_candidates
        ]

        broad_start_frame = find_stage5_broad_movement_boundary_frame(
            time_slice,
            ordered_records,
            search_start=time_slice.start_frame,
            search_end=start_search_end,
            prefer_latest=False,
        )
        broad_end_frame = find_stage5_broad_movement_boundary_frame(
            time_slice,
            ordered_records,
            search_start=end_search_start,
            search_end=time_slice.end_frame,
            prefer_latest=True,
        )
        stage5_debug['start_broad_movement_frame'] = broad_start_frame
        stage5_debug['start_broad_movement_time'] = None if broad_start_frame is None else Timecode(total_frames=broad_start_frame).to_hhmmssff()
        stage5_debug['end_broad_movement_frame'] = broad_end_frame
        stage5_debug['end_broad_movement_time'] = None if broad_end_frame is None else Timecode(total_frames=broad_end_frame).to_hhmmssff()

        local_start_frame, start_candidate_outcomes = find_stage5_valid_slice_start_trim_frame(
            time_slice,
            screened_candidate_union,
            ordered_records,
            sampled_frames,
            settings,
            cv2,
        )
        local_end_frame, end_candidate_outcomes = find_stage5_valid_slice_end_trim_frame(
            time_slice,
            screened_candidate_union,
            ordered_records,
            sampled_frames,
            settings,
            cv2,
        )
        stage5_debug['start_candidate_outcomes'] = start_candidate_outcomes
        stage5_debug['end_candidate_outcomes'] = end_candidate_outcomes
        stage5_debug['start_local_candidate_frame'] = local_start_frame
        stage5_debug['start_local_candidate_time'] = None if local_start_frame is None else Timecode(total_frames=local_start_frame).to_hhmmssff()
        stage5_debug['end_local_candidate_frame'] = local_end_frame
        stage5_debug['end_local_candidate_time'] = None if local_end_frame is None else Timecode(total_frames=local_end_frame).to_hhmmssff()

        trimmed_start_frame = None
        if broad_start_frame is not None:
            if (broad_start_frame - time_slice.start_frame) > STAGE5_SKIP_TRIM_NEAR_EDGE_FRAMES:
                trimmed_start_frame = broad_start_frame
                stage5_debug['start_trim_reason'] = 'broad_movement_start'
                stage5_debug['start_winning_path'] = 'broad_movement'
            else:
                stage5_debug['start_trim_reason'] = 'within_skip_guard'
                stage5_debug['start_winning_path'] = 'broad_movement'
        if trimmed_start_frame is None and local_start_frame is not None:
            if (local_start_frame - time_slice.start_frame) > STAGE5_SKIP_TRIM_NEAR_EDGE_FRAMES:
                trimmed_start_frame = local_start_frame
                stage5_debug['start_trim_reason'] = 'changed_cell_confirmation'
                stage5_debug['start_winning_path'] = 'changed_cell_confirmation'
            elif stage5_debug['start_trim_reason'] == 'no_trim':
                stage5_debug['start_trim_reason'] = 'within_skip_guard'
                stage5_debug['start_winning_path'] = 'changed_cell_confirmation'
        elif trimmed_start_frame is None and stage5_debug['start_trim_reason'] == 'no_trim':
            stage5_debug['start_trim_reason'] = 'no_qualified_start_candidate'
        candidate_slice = time_slice
        if (
            trimmed_start_frame is not None
            and trimmed_start_frame < candidate_slice.end_frame
            and stage5_trim_respects_minimum_clip_length(
                start_frame=trimmed_start_frame,
                end_frame=candidate_slice.end_frame,
                settings=settings,
            )
        ):
            candidate_slice = build_stage5_trimmed_time_slice(
                candidate_slice,
                start_frame=trimmed_start_frame,
                end_frame=candidate_slice.end_frame,
                ordered_records=ordered_records,
                stage5_debug=stage5_debug,
            )
            stage5_debug['start_trim_applied'] = True
            stage5_debug['trimmed_start_frame'] = trimmed_start_frame
            stage5_debug['trimmed_start_time'] = Timecode(total_frames=trimmed_start_frame).to_hhmmssff()
        elif trimmed_start_frame is not None and not stage5_trim_respects_minimum_clip_length(
            start_frame=trimmed_start_frame,
            end_frame=candidate_slice.end_frame,
            settings=settings,
        ):
            stage5_debug['start_trim_reason'] = 'minimum_clip_length_veto'
            stage5_debug['start_winning_path'] = 'veto'

        trimmed_end_frame = None
        if broad_end_frame is not None:
            proposed_broad_end_frame = min(candidate_slice.end_frame, broad_end_frame + 1)
            if (candidate_slice.end_frame - proposed_broad_end_frame) > STAGE5_SKIP_TRIM_NEAR_EDGE_FRAMES:
                trimmed_end_frame = proposed_broad_end_frame
                stage5_debug['end_trim_reason'] = 'broad_movement_end'
                stage5_debug['end_winning_path'] = 'broad_movement'
            else:
                stage5_debug['end_trim_reason'] = 'within_skip_guard'
                stage5_debug['end_winning_path'] = 'broad_movement'
        if trimmed_end_frame is None and local_end_frame is not None:
            if (candidate_slice.end_frame - local_end_frame) > STAGE5_SKIP_TRIM_NEAR_EDGE_FRAMES:
                trimmed_end_frame = local_end_frame
                stage5_debug['end_trim_reason'] = 'changed_cell_confirmation'
                stage5_debug['end_winning_path'] = 'changed_cell_confirmation'
            elif stage5_debug['end_trim_reason'] == 'no_trim':
                stage5_debug['end_trim_reason'] = 'within_skip_guard'
                stage5_debug['end_winning_path'] = 'changed_cell_confirmation'
        elif trimmed_end_frame is None and stage5_debug['end_trim_reason'] == 'no_trim':
            stage5_debug['end_trim_reason'] = 'no_qualified_end_candidate'
        if (
            trimmed_end_frame is not None
            and candidate_slice.start_frame < trimmed_end_frame
            and stage5_trim_respects_minimum_clip_length(
                start_frame=candidate_slice.start_frame,
                end_frame=trimmed_end_frame,
                settings=settings,
            )
        ):
            candidate_slice = build_stage5_trimmed_time_slice(
                candidate_slice,
                start_frame=candidate_slice.start_frame,
                end_frame=trimmed_end_frame,
                ordered_records=ordered_records,
                stage5_debug=stage5_debug,
            )
            stage5_debug['end_trim_applied'] = True
            stage5_debug['trimmed_end_frame'] = trimmed_end_frame
            stage5_debug['trimmed_end_time'] = Timecode(total_frames=trimmed_end_frame).to_hhmmssff()
        elif trimmed_end_frame is not None and not stage5_trim_respects_minimum_clip_length(
            start_frame=candidate_slice.start_frame,
            end_frame=trimmed_end_frame,
            settings=settings,
        ):
            stage5_debug['end_trim_reason'] = 'minimum_clip_length_veto'
            stage5_debug['end_winning_path'] = 'veto'

        if candidate_slice.stage5_debug is None:
            candidate_slice = replace(candidate_slice, stage5_debug=stage5_debug)

        refined_slices.append(candidate_slice)

    return refined_slices

def is_stage6_candidate_slice(
    time_slice: ClassifiedTimeSlice,
    maximum_boundary_frames: int = STAGE5_MIN_SUBDIVISION_FRAMES,
) -> bool:
    if time_slice.classification == 'valid':
        return True
    if time_slice.classification != 'boundary':
        return False
    if time_slice.reason != 'minimum_subdivision_size_reached':
        return False
    return (time_slice.end_frame - time_slice.start_frame) <= maximum_boundary_frames





def build_stage6_candidate_groups(
    refined_slices: Iterable[ClassifiedTimeSlice],
    maximum_boundary_frames: int = STAGE5_MIN_SUBDIVISION_FRAMES,
    merge_gap_frames: int = STAGE5_MIN_SUBDIVISION_FRAMES,
) -> list[list[ClassifiedTimeSlice]]:
    ordered_slices = [
        slice_info
        for slice_info in sorted(
            refined_slices,
            key=lambda slice_info: (slice_info.start_frame, slice_info.end_frame, slice_info.parent_union_index),
        )
        if is_stage6_candidate_slice(slice_info, maximum_boundary_frames=maximum_boundary_frames)
    ]
    if not ordered_slices:
        return []

    candidate_groups: list[list[ClassifiedTimeSlice]] = []
    current_group: list[ClassifiedTimeSlice] = [ordered_slices[0]]
    for next_slice in ordered_slices[1:]:
        previous_slice = current_group[-1]
        gap_frames = next_slice.start_frame - previous_slice.end_frame
        close_enough = gap_frames <= merge_gap_frames
        if close_enough:
            current_group.append(next_slice)
        else:
            candidate_groups.append(current_group)
            current_group = [next_slice]
    candidate_groups.append(current_group)
    return candidate_groups


def should_keep_stage6_group(candidate_group: list[ClassifiedTimeSlice]) -> bool:
    if not candidate_group:
        return False
    return any(slice_info.classification == 'valid' for slice_info in candidate_group)


def assemble_stage6_candidate_ranges(
    refined_slices: Iterable[ClassifiedTimeSlice],
    boundary_max_frames: int = STAGE5_MIN_SUBDIVISION_FRAMES,
    merge_gap_frames: int = STAGE5_MIN_SUBDIVISION_FRAMES,
) -> list[FinalCandidateRange]:
    candidate_groups = build_stage6_candidate_groups(
        refined_slices,
        maximum_boundary_frames=boundary_max_frames,
        merge_gap_frames=merge_gap_frames,
    )
    kept_groups = [candidate_group for candidate_group in candidate_groups if should_keep_stage6_group(candidate_group)]
    if not kept_groups:
        return []

    final_ranges: list[FinalCandidateRange] = []
    for range_index, candidate_group in enumerate(kept_groups, start=1):
        start_frame = candidate_group[0].start_frame
        end_frame = candidate_group[-1].end_frame
        source_classifications = tuple(dict.fromkeys(slice_info.classification for slice_info in candidate_group))
        boundary_count = sum(1 for slice_info in candidate_group if slice_info.classification == 'boundary')
        final_ranges.append(
            FinalCandidateRange(
                range_index=range_index,
                parent_union_index=candidate_group[0].parent_union_index,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=Timecode(total_frames=start_frame).to_hhmmssff(),
                end_time=Timecode(total_frames=end_frame).to_hhmmssff(),
                source_classifications=source_classifications,
                includes_boundary=boundary_count > 0,
                boundary_count=boundary_count,
            )
        )
    return final_ranges

# ============================================================
# SECTION J - Staged Pipeline Execution And Debug Output
# ============================================================

import json
import time


DEBUG_ARTIFACT_TITLES = {
    'movement_evidence_records': 'Stage 1A - Movement Evidence Record',
    'movement_spans': 'Stage 1B - Movement Spans',
    'reusable_stage3_art_state_samples': 'Stage 1C - Reusable Stage 2B Frame Payload',
    'candidate_unions': 'Stage 2A - Candidate Union Record',
    'screened_candidate_unions': 'Stage 3A - Union Screening',
    'stage3_screening_traces': 'Stage 3B - Screening Trace [Step 1 + Snapshot Rescue]',
    'classified_time_slices': 'Stage 4 - Probe-Band Classifications',
    'stage4_subregion_debug': 'Stage 4B - Probe Debug Output',
    'stage4_cell_reference_debug': 'Stage 4C - Per-Cell Reference Debug',
    'refined_sub_slices': 'Stage 5 - Local Boundary Refinement Output',
    'final_candidate_ranges': 'Stage 6A - Candidate Ranges [Pre-Filters]',
    'candidate_clips': 'Stage 6B - Candidate Pre-Clips [Post-Filters]',
}
DEBUG_SUMMARY_TITLE = 'Debug Summary'

def format_stage_elapsed(elapsed_seconds: float) -> str:
    whole_seconds = max(0, int(elapsed_seconds))
    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    remaining_seconds = whole_seconds % 60
    if hours > 0:
        return f'{hours:02}:{minutes:02}:{remaining_seconds:02}'
    return f'{minutes:02}:{remaining_seconds:02}'


def emit_stage_status(status_callback: Callable[[str], None] | None, message: str) -> None:
    if status_callback is not None:
        status_callback(message)


def build_stage_timing_entry(stage_key: str, stage_label: str, elapsed_seconds: float, item_count: int | None = None) -> dict[str, object]:
    entry: dict[str, object] = {
        'stage_key': stage_key,
        'stage_label': stage_label,
        'elapsed_seconds': round(elapsed_seconds, 6),
        'elapsed_hhmmss': format_stage_elapsed(elapsed_seconds),
    }
    if item_count is not None:
        entry['item_count'] = item_count
    return entry


def format_grid_coordinate_label(coordinate: GridCoordinate) -> str:
    row_index, column_index = coordinate
    column_label = chr(ord('A') + column_index)
    return f"{column_label}{row_index + 1}"



def serialize_grid_coordinate(coordinate: GridCoordinate) -> dict[str, object]:
    return {
        'row_index': coordinate[0],
        'column_index': coordinate[1],
        'label': format_grid_coordinate_label(coordinate),
    }


def serialize_movement_evidence_record(record: MovementEvidenceRecord) -> dict[str, object]:
    return {
        'record_index': record.record_index,
        'evaluation_point_timecode': record.evaluation_point_timecode,
        'frame_index': record.frame_index,
        'movement_present': record.movement_present,
        'touched_grid_coordinates': [list(coordinate) for coordinate in record.touched_grid_coordinates],
        'touched_grid_coordinate_labels': [format_grid_coordinate_label(coordinate) for coordinate in record.touched_grid_coordinates],
        'touched_grid_coordinate_details': [serialize_grid_coordinate(coordinate) for coordinate in record.touched_grid_coordinates],
        'touched_grid_coordinate_count': record.touched_grid_coordinate_count,
        'change_magnitude_score': round(record.change_magnitude_score, 6),
        'spatial_extent_score': round(record.spatial_extent_score, 6),
        'temporal_persistence_score': round(record.temporal_persistence_score, 6),
        'movement_strength_score': round(record.movement_strength_score, 6),
        'opening_signal': record.opening_signal,
        'continuation_signal': record.continuation_signal,
        'weak_signal': record.weak_signal,
    }





def build_precomputed_movement_evidence_cache_path(path: Path) -> Path:
    return path.with_suffix('.npz')
def write_precomputed_movement_evidence_record_cache(path: Path, serialized_records: list[dict[str, object]]) -> Path:
    output_path = build_precomputed_movement_evidence_cache_path(path)
    record_count = len(serialized_records)
    record_indices = np.empty(record_count, dtype=np.int32)
    frame_indices = np.empty(record_count, dtype=np.int32)
    movement_present = np.empty(record_count, dtype=np.bool_)
    opening_signal = np.empty(record_count, dtype=np.bool_)
    continuation_signal = np.empty(record_count, dtype=np.bool_)
    weak_signal = np.empty(record_count, dtype=np.bool_)
    change_magnitude_scores = np.empty(record_count, dtype=np.float32)
    spatial_extent_scores = np.empty(record_count, dtype=np.float32)
    temporal_persistence_scores = np.empty(record_count, dtype=np.float32)
    movement_strength_scores = np.empty(record_count, dtype=np.float32)
    touched_coordinate_offsets = np.zeros(record_count + 1, dtype=np.int32)
    flattened_touched_coordinates: list[tuple[int, int]] = []
    for index, payload in enumerate(serialized_records):
        record_indices[index] = int(payload['record_index'])
        frame_indices[index] = int(payload['frame_index'])
        movement_present[index] = bool(payload['movement_present'])
        opening_signal[index] = bool(payload.get('opening_signal', False))
        continuation_signal[index] = bool(payload.get('continuation_signal', False))
        weak_signal[index] = bool(payload.get('weak_signal', False))
        change_magnitude_scores[index] = float(payload['change_magnitude_score'])
        spatial_extent_scores[index] = float(payload['spatial_extent_score'])
        temporal_persistence_scores[index] = float(payload['temporal_persistence_score'])
        movement_strength_scores[index] = float(payload['movement_strength_score'])
        coordinates = [
            (int(coordinate[0]), int(coordinate[1]))
            for coordinate in payload.get('touched_grid_coordinates', [])
        ]
        flattened_touched_coordinates.extend(coordinates)
        touched_coordinate_offsets[index + 1] = len(flattened_touched_coordinates)
    touched_coordinate_values = (
        np.asarray(flattened_touched_coordinates, dtype=np.int16)
        if flattened_touched_coordinates
        else np.empty((0, 2), dtype=np.int16)
    )
    np.savez(
        output_path,
        record_indices=record_indices,
        frame_indices=frame_indices,
        movement_present=movement_present,
        opening_signal=opening_signal,
        continuation_signal=continuation_signal,
        weak_signal=weak_signal,
        change_magnitude_scores=change_magnitude_scores,
        spatial_extent_scores=spatial_extent_scores,
        temporal_persistence_scores=temporal_persistence_scores,
        movement_strength_scores=movement_strength_scores,
        touched_coordinate_offsets=touched_coordinate_offsets,
        touched_coordinate_values=touched_coordinate_values,
    )
    return output_path
def load_precomputed_movement_evidence_record_cache(path: Path) -> list[MovementEvidenceRecord]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            record_indices = payload['record_indices']
            frame_indices = payload['frame_indices']
            movement_present = payload['movement_present']
            opening_signal = payload['opening_signal']
            continuation_signal = payload['continuation_signal']
            weak_signal = payload['weak_signal']
            change_magnitude_scores = payload['change_magnitude_scores']
            spatial_extent_scores = payload['spatial_extent_scores']
            temporal_persistence_scores = payload['temporal_persistence_scores']
            movement_strength_scores = payload['movement_strength_scores']
            touched_coordinate_offsets = payload['touched_coordinate_offsets']
            touched_coordinate_values = payload['touched_coordinate_values']
    except Exception as exc:
        raise RuntimeError(f'Unable to load precomputed movement evidence cache: {path}') from exc
    record_count = len(record_indices)
    expected_lengths = (
        len(frame_indices),
        len(movement_present),
        len(opening_signal),
        len(continuation_signal),
        len(weak_signal),
        len(change_magnitude_scores),
        len(spatial_extent_scores),
        len(temporal_persistence_scores),
        len(movement_strength_scores),
    )
    if any(length != record_count for length in expected_lengths):
        raise RuntimeError(f'Precomputed movement evidence cache is inconsistent: {path}')
    if len(touched_coordinate_offsets) != record_count + 1:
        raise RuntimeError(f'Precomputed movement evidence cache has invalid touch offsets: {path}')
    if touched_coordinate_values.ndim != 2 or touched_coordinate_values.shape[1] != 2:
        raise RuntimeError(f'Precomputed movement evidence cache has invalid touch coordinates: {path}')
    records: list[MovementEvidenceRecord] = []
    for index in range(record_count):
        touch_start = int(touched_coordinate_offsets[index])
        touch_end = int(touched_coordinate_offsets[index + 1])
        if touch_start > touch_end or touch_end > len(touched_coordinate_values):
            raise RuntimeError(f'Precomputed movement evidence cache has invalid touch bounds: {path}')
        touched_grid_coordinates = tuple(
            (int(coordinate[0]), int(coordinate[1]))
            for coordinate in touched_coordinate_values[touch_start:touch_end]
        )
        frame_index = int(frame_indices[index])
        records.append(
            MovementEvidenceRecord(
                record_index=int(record_indices[index]),
                evaluation_point_timecode=Timecode(total_frames=frame_index).to_hhmmssff(),
                frame_index=frame_index,
                movement_present=bool(movement_present[index]),
                touched_grid_coordinates=touched_grid_coordinates,
                touched_grid_coordinate_count=len(touched_grid_coordinates),
                change_magnitude_score=float(change_magnitude_scores[index]),
                spatial_extent_score=float(spatial_extent_scores[index]),
                temporal_persistence_score=float(temporal_persistence_scores[index]),
                movement_strength_score=float(movement_strength_scores[index]),
                opening_signal=bool(opening_signal[index]),
                continuation_signal=bool(continuation_signal[index]),
                weak_signal=bool(weak_signal[index]),
            )
        )
    return records
def find_precomputed_movement_evidence_cache_path(path: Path) -> Path | None:
    cache_path = build_precomputed_movement_evidence_cache_path(path)
    if not cache_path.exists():
        return None
    return cache_path
def deserialize_movement_evidence_record(payload: dict[str, object]) -> MovementEvidenceRecord:
    touched_grid_coordinates = tuple(
        (int(coordinate[0]), int(coordinate[1]))
        for coordinate in payload.get('touched_grid_coordinates', [])
    )
    return MovementEvidenceRecord(
        record_index=int(payload['record_index']),
        evaluation_point_timecode=str(payload['evaluation_point_timecode']),
        frame_index=int(payload['frame_index']),
        movement_present=bool(payload['movement_present']),
        touched_grid_coordinates=touched_grid_coordinates,
        touched_grid_coordinate_count=int(payload.get('touched_grid_coordinate_count', len(touched_grid_coordinates))),
        change_magnitude_score=float(payload['change_magnitude_score']),
        spatial_extent_score=float(payload['spatial_extent_score']),
        temporal_persistence_score=float(payload['temporal_persistence_score']),
        movement_strength_score=float(payload['movement_strength_score']),
        opening_signal=bool(payload.get('opening_signal', False)),
        continuation_signal=bool(payload.get('continuation_signal', False)),
        weak_signal=bool(payload.get('weak_signal', False)),
    )


def load_precomputed_movement_evidence_records(path: Path) -> list[MovementEvidenceRecord]:
    cache_path = find_precomputed_movement_evidence_cache_path(path)
    if cache_path is not None:
        try:
            return load_precomputed_movement_evidence_record_cache(cache_path)
        except RuntimeError:
            pass
    raw_payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(raw_payload, list):
        raise RuntimeError(f'Precomputed movement evidence JSON must contain a top-level list: {path}')
    return [deserialize_movement_evidence_record(item) for item in raw_payload]


def infer_debug_stem_from_artifact_path(artifact_path: Path, artifact_key: str, suffix: str) -> Path | None:
    artifact_title = DEBUG_ARTIFACT_TITLES[artifact_key]
    artifact_suffix = f' - {artifact_title}{suffix}'
    if not artifact_path.name.endswith(artifact_suffix):
        return None
    return artifact_path.with_name(artifact_path.name[:-len(artifact_suffix)])


def get_stage3_canvas_shape(sample: dict[str, object]) -> tuple[int, int]:
    canvas_shape = sample.get('canvas_shape')
    if canvas_shape is not None:
        return (int(canvas_shape[0]), int(canvas_shape[1]))
    gray = sample.get('gray')
    if gray is not None:
        return tuple(int(dimension) for dimension in gray.shape)
    art_gray = sample.get('art_gray')
    if art_gray is not None:
        return tuple(int(dimension) for dimension in art_gray.shape)
    raise RuntimeError('Stage 3 sample is missing canvas shape metadata.')


def write_reusable_stage3_art_state_sample_cache(debug_stem: Path, sampled_frames: list[dict[str, object]]) -> Path:
    output_path = build_staged_debug_output_path(debug_stem, 'reusable_stage3_art_state_samples', '.npz')
    frame_indices = np.asarray([int(sample['frame_index']) for sample in sampled_frames], dtype=np.int32)
    if sampled_frames:
        canvas_shape = np.asarray(get_stage3_canvas_shape(sampled_frames[0]), dtype=np.int32)
        gray_stack = np.stack([np.asarray(get_stage3_sample_gray(sample), dtype=np.uint8) for sample in sampled_frames], axis=0)
    else:
        canvas_shape = np.asarray((0, 0), dtype=np.int32)
        gray_stack = np.empty((0, 0, 0), dtype=np.uint8)
    np.savez(output_path, frame_indices=frame_indices, canvas_shape=canvas_shape, gray=gray_stack)
    return output_path


def load_reusable_stage3_art_state_sample_cache(path: Path) -> list[dict[str, object]]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            frame_indices = payload['frame_indices']
            gray_stack = payload['gray'] if 'gray' in payload else payload['art_gray']
            if 'canvas_shape' in payload:
                canvas_shape = tuple(int(dimension) for dimension in payload['canvas_shape'].tolist())
            elif 'canvas_gray' in payload:
                canvas_gray_stack = payload['canvas_gray']
                if len(frame_indices) != len(canvas_gray_stack) or len(frame_indices) != len(gray_stack):
                    raise RuntimeError(f'Reusable Stage 2B frame payload is inconsistent: {path}')
                return [
                    {
                        'frame_index': int(frame_indices[index]),
                        'canvas_shape': tuple(int(dimension) for dimension in canvas_gray_stack[index].shape),
                        'gray': gray_stack[index],
                        'art_gray': gray_stack[index],
                    }
                    for index in range(len(frame_indices))
                ]
            else:
                raise RuntimeError(f'Reusable Stage 2B frame payload is missing canvas shape metadata: {path}')
    except Exception as exc:
        raise RuntimeError(f'Unable to load reusable Stage 2B frame payload: {path}') from exc
    if len(frame_indices) != len(gray_stack):
        raise RuntimeError(f'Reusable Stage 2B frame payload is inconsistent: {path}')
    return [
        {
            'frame_index': int(frame_indices[index]),
            'canvas_shape': canvas_shape,
            'gray': gray_stack[index],
            'art_gray': gray_stack[index],
        }
        for index in range(len(frame_indices))
    ]


def find_precomputed_stage3_art_state_sample_cache_path_from_movement_evidence_path(movement_evidence_path: Path) -> Path | None:
    debug_stem = infer_debug_stem_from_artifact_path(movement_evidence_path, 'movement_evidence_records', '.json')
    if debug_stem is None:
        return None
    cache_path = build_staged_debug_output_path(debug_stem, 'reusable_stage3_art_state_samples', '.npz')
    if not cache_path.exists():
        return None
    return cache_path


def load_precomputed_stage3_art_state_sample_cache_from_movement_evidence_path(
    movement_evidence_path: Path,
) -> list[dict[str, object]] | None:
    cache_path = find_precomputed_stage3_art_state_sample_cache_path_from_movement_evidence_path(movement_evidence_path)
    if cache_path is None:
        return None
    return load_reusable_stage3_art_state_sample_cache(cache_path)


def serialize_movement_span(span: MovementSpan) -> dict[str, object]:
    return {
        'span_index': span.span_index,
        'start_frame': span.start_frame,
        'end_frame': span.end_frame,
        'start_time': span.start_time,
        'end_time': span.end_time,
        'footprint': [list(coordinate) for coordinate in sorted(span.footprint)],
        'footprint_labels': [format_grid_coordinate_label(coordinate) for coordinate in sorted(span.footprint)],
        'footprint_details': [serialize_grid_coordinate(coordinate) for coordinate in sorted(span.footprint)],
        'footprint_size': span.footprint_size,
        'record_indices': list(span.record_indices),
    }



def serialize_candidate_union(candidate_union: CandidateUnion) -> dict[str, object]:
    return {
        'union_index': candidate_union.union_index,
        'start_frame': candidate_union.start_frame,
        'end_frame': candidate_union.end_frame,
        'start_time': candidate_union.start_time,
        'end_time': candidate_union.end_time,
        'member_movement_span_indices': [span.span_index for span in candidate_union.member_movement_spans],
        'union_footprint': [list(coordinate) for coordinate in sorted(candidate_union.union_footprint)],
        'union_footprint_labels': [format_grid_coordinate_label(coordinate) for coordinate in sorted(candidate_union.union_footprint)],
        'union_footprint_details': [serialize_grid_coordinate(coordinate) for coordinate in sorted(candidate_union.union_footprint)],
        'union_footprint_size': candidate_union.union_footprint_size,
    }



def serialize_screened_candidate_union(screened_union: ScreenedCandidateUnion) -> dict[str, object]:
    return {
        'candidate_union_index': screened_union.candidate_union.union_index,
        'screening_result': screened_union.screening_result,
        'surviving': screened_union.surviving,
        'provisional_survival': screened_union.provisional_survival,
        'reason': screened_union.reason,
        'within_union_record_count': screened_union.within_union_record_count,
        'before_record_count': screened_union.before_record_count,
        'after_record_count': screened_union.after_record_count,
        'mean_movement_strength': round(screened_union.mean_movement_strength, 6),
        'mean_temporal_persistence': round(screened_union.mean_temporal_persistence, 6),
        'mean_spatial_extent': round(screened_union.mean_spatial_extent, 6),
        'lasting_change_evidence_score': round(screened_union.lasting_change_evidence_score, 6),
        'before_reference_activity': round(screened_union.before_reference_activity, 6),
        'after_reference_activity': round(screened_union.after_reference_activity, 6),
        'reference_windows_reliable': screened_union.reference_windows_reliable,
        'stage3_mode': screened_union.stage3_mode,
        'stage3_alignment_mode': screened_union.stage3_alignment_mode,
        'stage3_persistent_difference_score': round(screened_union.stage3_persistent_difference_score, 6),
        'stage3_footprint_support_score': round(screened_union.stage3_footprint_support_score, 6),
        'stage3_after_window_persistence_score': round(screened_union.stage3_after_window_persistence_score, 6),
        'stage3_before_window_start': screened_union.stage3_before_window_start,
        'stage3_before_window_end': screened_union.stage3_before_window_end,
        'stage3_after_window_start': screened_union.stage3_after_window_start,
        'stage3_after_window_end': screened_union.stage3_after_window_end,
        'stage3_before_sample_count': screened_union.stage3_before_sample_count,
        'stage3_after_sample_count': screened_union.stage3_after_sample_count,
        'stage3_reveal_sample_count': screened_union.stage3_reveal_sample_count,
        'stage3_before_window_quality_score': round(screened_union.stage3_before_window_quality_score, 6),
        'stage3_after_window_quality_score': round(screened_union.stage3_after_window_quality_score, 6),
        'stage3_reveal_window_quality_score': round(screened_union.stage3_reveal_window_quality_score, 6),
        'stage3_before_window_candidate_count': screened_union.stage3_before_window_candidate_count,
        'stage3_after_window_candidate_count': screened_union.stage3_after_window_candidate_count,
        'stage3_reveal_window_candidate_count': screened_union.stage3_reveal_window_candidate_count,
        'stage3_before_window_tier': screened_union.stage3_before_window_tier,
        'stage3_after_window_tier': screened_union.stage3_after_window_tier,
        'stage3_reveal_window_tier': screened_union.stage3_reveal_window_tier,
        'stage3_reveal_window_start': screened_union.stage3_reveal_window_start,
        'stage3_reveal_window_end': screened_union.stage3_reveal_window_end,
        'stage3_reveal_window_hold_score': round(screened_union.stage3_reveal_window_hold_score, 6),
    }



def serialize_stage3_screening_trace(screened_union: ScreenedCandidateUnion) -> dict[str, object]:
    if screened_union.stage3_debug_trace is None:
        return {
            'candidate_union_index': screened_union.candidate_union.union_index,
            'final_stage3_outcome': {
                'screening_result': screened_union.screening_result,
                'surviving': screened_union.surviving,
                'reason': screened_union.reason,
                'mode': screened_union.stage3_mode,
                'alignment_mode': screened_union.stage3_alignment_mode,
            },
        }
    return screened_union.stage3_debug_trace



def serialize_classified_time_slice(time_slice: ClassifiedTimeSlice) -> dict[str, object]:
    return {
        'slice_index': time_slice.slice_index,
        'parent_union_index': time_slice.parent_union_index,
        'slice_level': time_slice.slice_level,
        'start_frame': time_slice.start_frame,
        'end_frame': time_slice.end_frame,
        'start_time': time_slice.start_time,
        'end_time': time_slice.end_time,
        'footprint': [list(coordinate) for coordinate in sorted(time_slice.footprint)],
        'footprint_labels': [format_grid_coordinate_label(coordinate) for coordinate in sorted(time_slice.footprint)],
        'footprint_details': [serialize_grid_coordinate(coordinate) for coordinate in sorted(time_slice.footprint)],
        'footprint_size': time_slice.footprint_size,
        'within_slice_record_count': time_slice.within_slice_record_count,
        'classification': time_slice.classification,
        'reason': time_slice.reason,
        'lasting_change_evidence_score': round(time_slice.lasting_change_evidence_score, 6),
        'before_reference_activity': round(time_slice.before_reference_activity, 6),
        'after_reference_activity': round(time_slice.after_reference_activity, 6),
        'reference_windows_reliable': time_slice.reference_windows_reliable,
        'parent_range': list(time_slice.parent_range) if time_slice.parent_range is not None else None,
        'stage5_debug': time_slice.stage5_debug,
    }



def serialize_stage4_subregion_result(subregion_result: dict[str, object]) -> dict[str, object]:
    subregion = frozenset(subregion_result.get('subregion', []))
    return {
        'subregion': [list(coordinate) for coordinate in sorted(subregion)],
        'subregion_labels': [format_grid_coordinate_label(coordinate) for coordinate in sorted(subregion)],
        'subregion_details': [serialize_grid_coordinate(coordinate) for coordinate in sorted(subregion)],
        'subregion_size': len(subregion),
        'comparison_state': str(subregion_result.get('comparison_state', 'reference_missing')),
        'settled': bool(subregion_result.get('settled', False)),
        'supported': bool(subregion_result.get('supported', False)),
        'unresolved': bool(subregion_result.get('unresolved', False)),
        'reference_windows_reliable': bool(subregion_result.get('reference_windows_reliable', False)),
        'meaningful_unsettled_activity': bool(subregion_result.get('meaningful_unsettled_activity', False)),
        'before_reference_activity': round(float(subregion_result.get('before_reference_activity', 0.0)), 6),
        'after_reference_activity': round(float(subregion_result.get('after_reference_activity', 0.0)), 6),
        'evidence_score': round(float(subregion_result.get('evidence_score', 0.0)), 6),
        'persistent_difference_score': round(float(subregion_result.get('persistent_difference_score', 0.0)), 6),
        'footprint_support_score': round(float(subregion_result.get('footprint_support_score', 0.0)), 6),
        'after_window_persistence_score': round(float(subregion_result.get('after_window_persistence_score', 0.0)), 6),
        'before_window_candidate_count': int(subregion_result.get('before_window_candidate_count', 0)),
        'after_window_candidate_count': int(subregion_result.get('after_window_candidate_count', 0)),
    }



def serialize_stage4_time_slice_subregion_debug_entry(
    time_slice: ClassifiedTimeSlice,
    evaluation: dict[str, object],
) -> dict[str, object]:
    summary = evaluation.get('summary')
    serialized_summary = None
    if isinstance(summary, dict):
        serialized_summary = {
            'changed_footprint': int(summary.get('changed_footprint', 0)),
            'unchanged_footprint': int(summary.get('unchanged_footprint', 0)),
            'unsettled_footprint': int(summary.get('unsettled_footprint', 0)),
            'ambiguous_footprint': int(summary.get('ambiguous_footprint', 0)),
            'settled_footprint': int(summary.get('settled_footprint', 0)),
            'changed_ratio': round(float(summary.get('changed_ratio', 0.0)), 6),
            'unsettled_ratio': round(float(summary.get('unsettled_ratio', 0.0)), 6),
            'ambiguous_ratio': round(float(summary.get('ambiguous_ratio', 0.0)), 6),
            'settled_evidence_score': round(float(summary.get('settled_evidence_score', 0.0)), 6),
        }

    subregions = [frozenset(subregion) for subregion in evaluation.get('subregions', [])]
    return {
        **serialize_classified_time_slice(time_slice),
        'subregion_count': len(subregions),
        'subregions': [
            {
                'subregion': [list(coordinate) for coordinate in sorted(subregion)],
                'subregion_labels': [format_grid_coordinate_label(coordinate) for coordinate in sorted(subregion)],
                'subregion_details': [serialize_grid_coordinate(coordinate) for coordinate in sorted(subregion)],
                'subregion_size': len(subregion),
            }
            for subregion in subregions
        ],
        'subregion_summary': serialized_summary,
        'subregion_results': [
            serialize_stage4_subregion_result(subregion_result)
            for subregion_result in evaluation.get('subregion_results', [])
        ],
        'probe_model': evaluation.get('probe_model'),
        'probe_anchor_time': evaluation.get('probe_anchor_time'),
        'probe_local_evaluation_window': evaluation.get('probe_local_evaluation_window'),
        'recently_active_cells': evaluation.get('recently_active_cells', []),
        'judgeable_cells': evaluation.get('judgeable_cells', []),
        'contaminated_cells': evaluation.get('contaminated_cells', []),
        'post_reset_contaminated_cells': evaluation.get('post_reset_contaminated_cells', []),
        'confirmed_changed_cells': evaluation.get('confirmed_changed_cells', []),
        'confirmed_unchanged_cells': evaluation.get('confirmed_unchanged_cells', []),
        'unresolved_cells': evaluation.get('unresolved_cells', []),
        'cell_results': evaluation.get('cell_results', []),
        'reason_summary': evaluation.get('reason_summary'),
        'relevant_cell_count': int(evaluation.get('relevant_cell_count', 0)),
        'judgeable_cell_count': int(evaluation.get('judgeable_cell_count', 0)),
        'confirmed_changed_cell_count': int(evaluation.get('confirmed_changed_cell_count', 0)),
        'confirmed_unchanged_cell_count': int(evaluation.get('confirmed_unchanged_cell_count', 0)),
        'unresolved_cell_count': int(evaluation.get('unresolved_cell_count', 0)),
        'judgeable_coverage': round(float(evaluation.get('judgeable_coverage', 0.0)), 6),
        'changed_support_score': round(float(evaluation.get('changed_support_score', 0.0)), 6),
        'judgeable_changed_support_score': round(float(evaluation.get('judgeable_changed_support_score', 0.0)), 6),
        'unchanged_support_score': round(float(evaluation.get('unchanged_support_score', 0.0)), 6),
        'unresolved_support_score': round(float(evaluation.get('unresolved_support_score', 0.0)), 6),
        'structural_holding_support': bool(evaluation.get('structural_holding_support', False)),
        'opening_zone_low_confidence': bool(evaluation.get('opening_zone_low_confidence', False)),
        'opening_attribution_start_frame': evaluation.get('opening_attribution_start_frame'),
        'opening_attribution_start_time': evaluation.get('opening_attribution_start_time'),
        'late_resolution_only': bool(evaluation.get('late_resolution_only', False)),
        'state_reset_candidate': bool(evaluation.get('state_reset_candidate', False)),
        'changed_touch_frame_count': int(evaluation.get('changed_touch_frame_count', 0)),
        'changed_touch_frame_ratio': round(float(evaluation.get('changed_touch_frame_ratio', 0.0)), 6),
        'probe_label': evaluation.get('probe_label'),
        'before_window_candidate_count': int(evaluation.get('before_window_candidate_count', 0)),
        'current_window_candidate_count': int(evaluation.get('current_window_candidate_count', 0)),
        'baseline_assigned_cell_count': int(evaluation.get('baseline_assigned_cell_count', 0)),
        'baseline_comparison_update_cell_count': int(evaluation.get('baseline_comparison_update_cell_count', 0)),
        'current_window': evaluation.get('current_window'),
        'probe_timings': evaluation.get('probe_timings', {}),
        'stage4_union_baseline_assignment_seconds': round(float(evaluation.get('stage4_union_baseline_assignment_seconds', 0.0)), 6),
    }

def serialize_stage4_cell_reference_debug_entries(
    time_slice: ClassifiedTimeSlice,
    evaluation: dict[str, object],
) -> list[dict[str, object]]:
    serialized_probe = serialize_classified_time_slice(time_slice)
    probe_anchor_time = evaluation.get('probe_anchor_time')
    probe_local_evaluation_window = evaluation.get('probe_local_evaluation_window')
    return [
        {
            **serialized_probe,
            'probe_anchor_time': probe_anchor_time,
            'probe_local_evaluation_window': probe_local_evaluation_window,
            'probe_label': evaluation.get('probe_label'),
            **cell_entry,
        }
        for cell_entry in evaluation.get('cell_reference_debug', [])
        if isinstance(cell_entry, dict)
    ]


def serialize_final_candidate_range(final_range: FinalCandidateRange) -> dict[str, object]:
    return {
        'range_index': final_range.range_index,
        'parent_union_index': final_range.parent_union_index,
        'start_frame': final_range.start_frame,
        'end_frame': final_range.end_frame,
        'start_time': final_range.start_time,
        'end_time': final_range.end_time,
        'source_classifications': list(final_range.source_classifications),
        'includes_boundary': final_range.includes_boundary,
        'boundary_count': final_range.boundary_count,
    }



def summarize_slice_classifications(
    slices: Iterable[dict[str, object]],
) -> dict[str, int]:
    classification_counts = {
        'valid': 0,
        'invalid': 0,
        'undetermined': 0,
        'boundary': 0,
    }
    for slice_info in slices:
        classification = slice_info.get('classification')
        if classification in classification_counts:
            classification_counts[classification] += 1
    return classification_counts



def format_final_range_source_description(source_classifications: Iterable[object]) -> str:
    normalized_sources = {str(item) for item in source_classifications}
    if 'valid' in normalized_sources and 'boundary' in normalized_sources:
        return 'valid + boundary support'
    if normalized_sources == {'valid'}:
        return 'valid only'
    if normalized_sources == {'boundary'}:
        return 'boundary only'
    if not normalized_sources:
        return 'no source classifications recorded'
    return ' + '.join(sorted(normalized_sources))



def build_staged_debug_summary_lines(debug_payload: dict[str, object]) -> list[str]:
    records = debug_payload.get('movement_evidence_records', [])
    spans = debug_payload.get('movement_spans', [])
    candidate_unions = debug_payload.get('candidate_unions', [])
    screened_candidate_unions = debug_payload.get('screened_candidate_unions', [])
    classified_time_slices = debug_payload.get('classified_time_slices', [])
    refined_sub_slices = debug_payload.get('refined_sub_slices', [])
    final_candidate_ranges = debug_payload.get('final_candidate_ranges', [])
    candidate_clips = debug_payload.get('candidate_clips', [])
    stage_timings = debug_payload.get('stage_timings', [])

    surviving_union_count = sum(
        1 for screened_union in screened_candidate_unions
        if screened_union.get('surviving')
    )
    rejected_union_count = len(screened_candidate_unions) - surviving_union_count
    provisional_survival_count = sum(
        1 for screened_union in screened_candidate_unions
        if screened_union.get('provisional_survival')
    )
    stage4_counts = summarize_slice_classifications(classified_time_slices)
    stage5_counts = summarize_slice_classifications(refined_sub_slices)
    boundary_supported_range_count = sum(
        1 for final_range in final_candidate_ranges
        if final_range.get('includes_boundary')
    )

    summary_lines = [
        'Staged detector summary',
        '',
        'Movement evidence',
        f'- Movement evidence records created: {len(records)}',
        '',
        'Stage 1 - Movement spans',
        f'- Movement spans created: {len(spans)}',
    ]

    if spans:
        summary_lines.append('')
        summary_lines.append('Movement span ranges')
        for span in spans:
            summary_lines.append(
                f"- Span {span.get('span_index', '?')}: {span.get('start_time', 'unknown')} to {span.get('end_time', 'unknown')}"
            )

    summary_lines.extend([
        '',
        'Stage 2 - Candidate unions',
        f'- Candidate unions created: {len(candidate_unions)}',
    ])

    if candidate_unions:
        summary_lines.append('')
        summary_lines.append('Candidate union ranges')
        for candidate_union in candidate_unions:
            summary_lines.append(
                f"- Union {candidate_union.get('union_index', '?')}: {candidate_union.get('start_time', 'unknown')} to {candidate_union.get('end_time', 'unknown')}"
            )

    summary_lines.extend([
        '',
        'Stage 3 - Union screening',
        f'- Candidate unions screened: {len(screened_candidate_unions)}',
        f'- Survived: {surviving_union_count}',
        f'- Rejected: {rejected_union_count}',
        f'- Provisional survivals: {provisional_survival_count}',
        '',
        'Stage 4 - Probe-bands',
        f'- Probe-bands created: {len(classified_time_slices)}',
        f"- Valid: {stage4_counts['valid']}",
        f"- Invalid: {stage4_counts['invalid']}",
        f"- Undetermined: {stage4_counts['undetermined']}",
        '',
        'Stage 5 - Local boundary refinement',
        f'- Refined slices produced: {len(refined_sub_slices)}',
        f"- Valid: {stage5_counts['valid']}",
        f"- Invalid: {stage5_counts['invalid']}",
        f"- Boundary: {stage5_counts['boundary']}",
        f"- Undetermined remaining after refinement: {stage5_counts['undetermined']}",
        '',
        'Stage 6 - Final retained ranges',
        f'- Final retained ranges built from valid material: {len(final_candidate_ranges)}',
        f'- Ranges that include boundary support: {boundary_supported_range_count}',
        f'- Candidate clips produced: {len(candidate_clips)}',
    ])

    if stage_timings:
        summary_lines.append('')
        summary_lines.append('Runtime timings')
        for timing in stage_timings:
            stage_label = timing.get('stage_label', 'unknown stage')
            elapsed_hhmmss = timing.get('elapsed_hhmmss', 'unknown')
            item_count = timing.get('item_count')
            detail_suffix = '' if item_count is None else f' ({item_count} items)'
            summary_lines.append(f'- {stage_label}: {elapsed_hhmmss}{detail_suffix}')

    if final_candidate_ranges:
        summary_lines.append('')
        summary_lines.append('Final retained ranges')
        for final_range in final_candidate_ranges:
            summary_lines.append(
                f"- Range {final_range.get('range_index', '?')}: {final_range.get('start_time', 'unknown')} to {final_range.get('end_time', 'unknown')}"
            )
            summary_lines.append(
                f"  Built from: {format_final_range_source_description(final_range.get('source_classifications', []))}"
            )
            summary_lines.append(
                f"  Boundary slices attached: {final_range.get('boundary_count', 0)}"
            )

    if candidate_clips:
        summary_lines.append('')
        summary_lines.append('Candidate clips')
        for candidate_clip in candidate_clips:
            summary_lines.append(
                f"- Clip {candidate_clip.get('clip_index', '?')}: {candidate_clip.get('clip_start', 'unknown')} to {candidate_clip.get('clip_end', 'unknown')}"
            )
            summary_lines.append(
                f"  Activity inside clip: {candidate_clip.get('activity_start', 'unknown')} to {candidate_clip.get('activity_end', 'unknown')}"
            )

    return summary_lines



def detect_staged_activity_ranges(
    video_path: Path,
    chapter_range: ChapterRange,
    settings: DetectorSettings,
    progress_callback: Callable[[int], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
    debug_stem: Path | None = None,
    use_stage3_art_state_prototype: bool = False,
    precomputed_movement_evidence_path: Path | None = None,
) -> tuple[list[tuple[int, int]], dict[str, object]]:
    stage_timings: list[dict[str, object]] = []
    total_started_at = time.perf_counter()

    def flush_debug_payload(debug_payload: dict[str, object]) -> None:
        if debug_stem is None:
            return
        write_staged_debug_artifacts(debug_stem, debug_payload)

    if precomputed_movement_evidence_path is None:
        emit_stage_status(status_callback, 'Runtime Stage 1A - Scanning chapter for movement evidence started')
    else:
        emit_stage_status(status_callback, f"Runtime Stage 1A - Loading precomputed movement evidence started ({precomputed_movement_evidence_path})")
    stage_started_at = time.perf_counter()
    reused_stage3_sample_cache_path: Path | None = None
    if precomputed_movement_evidence_path is None:
        records, retained_stage3_samples = detect_movement_evidence_records(
            video_path=video_path,
            chapter_range=chapter_range,
            settings=settings,
            progress_callback=progress_callback,
        )
    else:
        records = load_precomputed_movement_evidence_records(precomputed_movement_evidence_path)
        reused_stage3_sample_cache_path = find_precomputed_stage3_art_state_sample_cache_path_from_movement_evidence_path(
            precomputed_movement_evidence_path
        )
        retained_stage3_samples = None
        if reused_stage3_sample_cache_path is not None:
            try:
                reused_stage3_sample_cache_size = reused_stage3_sample_cache_path.stat().st_size
            except OSError:
                reused_stage3_sample_cache_size = None
            if (
                reused_stage3_sample_cache_size is not None
                and reused_stage3_sample_cache_size > MAX_AUTOMATIC_REUSED_STAGE3_SAMPLE_CACHE_BYTES
            ):
                emit_stage_status(
                    status_callback,
                    'Runtime Stage 1C - Skipping reused Stage 2B frame payload auto-load '
                    f"({reused_stage3_sample_cache_size // (1024 * 1024)} MB cache); Stage 2B will rescan samples",
                )
                reused_stage3_sample_cache_path = None
            else:
                retained_stage3_samples = load_reusable_stage3_art_state_sample_cache(
                    reused_stage3_sample_cache_path
                )
    elapsed_seconds = time.perf_counter() - stage_started_at
    stage_timings.append(build_stage_timing_entry('runtime_stage_1a_movement_evidence', 'Runtime Stage 1A - Scanning chapter for movement evidence', elapsed_seconds, len(records)))
    if precomputed_movement_evidence_path is None:
        emit_stage_status(status_callback, f"Runtime Stage 1A - Scanning chapter for movement evidence complete in {format_stage_elapsed(elapsed_seconds)} ({len(records)} movement evidence records)")
    else:
        emit_stage_status(status_callback, f"Runtime Stage 1A - Loading precomputed movement evidence complete in {format_stage_elapsed(elapsed_seconds)} ({len(records)} movement evidence records)")
    serialized_records = [serialize_movement_evidence_record(record) for record in records]
    flush_debug_payload({
        'movement_evidence_records': serialized_records,
        'stage_timings': stage_timings,
    })

    emit_stage_status(status_callback, 'Runtime Stage 1B - Building movement spans started')
    stage_started_at = time.perf_counter()
    movement_spans = build_stage1_movement_spans(records, settings)
    elapsed_seconds = time.perf_counter() - stage_started_at
    stage_timings.append(build_stage_timing_entry('runtime_stage_1b_movement_spans', 'Runtime Stage 1B - Building movement spans', elapsed_seconds, len(movement_spans)))
    emit_stage_status(status_callback, f"Runtime Stage 1B - Building movement spans complete in {format_stage_elapsed(elapsed_seconds)} ({len(movement_spans)} movement spans)")
    serialized_movement_spans = [serialize_movement_span(span) for span in movement_spans]
    flush_debug_payload({
        'movement_evidence_records': serialized_records,
        'movement_spans': serialized_movement_spans,
        'stage_timings': stage_timings,
    })

    emit_stage_status(status_callback, 'Runtime Stage 2A - Building candidate unions started')
    stage_started_at = time.perf_counter()
    candidate_unions = build_stage2_candidate_unions(movement_spans)
    elapsed_seconds = time.perf_counter() - stage_started_at
    stage_timings.append(build_stage_timing_entry('runtime_stage_2a_candidate_unions', 'Runtime Stage 2A - Building candidate unions', elapsed_seconds, len(candidate_unions)))
    emit_stage_status(status_callback, f"Runtime Stage 2A - Building candidate unions complete in {format_stage_elapsed(elapsed_seconds)} ({len(candidate_unions)} candidate unions)")
    serialized_candidate_unions = [serialize_candidate_union(candidate_union) for candidate_union in candidate_unions]
    flush_debug_payload({
        'movement_evidence_records': serialized_records,
        'movement_spans': serialized_movement_spans,
        'candidate_unions': serialized_candidate_unions,
        'stage_timings': stage_timings,
    })

    reuse_source_description: str | None = None
    if retained_stage3_samples is not None:
        reuse_source_description = (
            'reusing in-memory Stage 1 sampled frames from this run'
            if precomputed_movement_evidence_path is None
            else 'reusing Stage 1C cached frame payload'
        )

    emit_stage_status(status_callback, 'Runtime Stage 2B - Collecting Stage 3 art-state samples started')
    stage_started_at = time.perf_counter()
    stage3_art_state_samples = collect_stage3_art_state_samples(
        video_path=video_path,
        chapter_range=chapter_range,
        settings=settings,
        status_callback=status_callback,
        precomputed_samples=retained_stage3_samples,
        reuse_source_description=reuse_source_description,
    )
    elapsed_seconds = time.perf_counter() - stage_started_at
    stage_timings.append(build_stage_timing_entry('runtime_stage_2b_art_state_samples', 'Runtime Stage 2B - Collecting Stage 3 art-state samples', elapsed_seconds, len(stage3_art_state_samples)))
    emit_stage_status(status_callback, f"Runtime Stage 2B - Collecting Stage 3 art-state samples complete in {format_stage_elapsed(elapsed_seconds)} ({len(stage3_art_state_samples)} sampled frames)")
    if debug_stem is not None:
        reusable_sample_cache_output_path = build_staged_debug_output_path(debug_stem, 'reusable_stage3_art_state_samples', '.npz')
        if not ENABLE_REUSABLE_STAGE3_SAMPLE_CACHE_WRITES:
            if reused_stage3_sample_cache_path is not None:
                emit_stage_status(
                    status_callback,
                    'Runtime Stage 1C - Reusable Stage 2B frame payload write skipped (temporarily disabled during benchmarking reruns)',
                )
            else:
                emit_stage_status(
                    status_callback,
                    'Runtime Stage 1C - Reusable Stage 2B frame payload write skipped (temporarily disabled)',
                )
        elif reused_stage3_sample_cache_path is not None:
            emit_stage_status(status_callback, 'Runtime Stage 1C - Linking reused Stage 2B frame payload into current debug folder started')
            stage_started_at = time.perf_counter()
            if reusable_sample_cache_output_path != reused_stage3_sample_cache_path and not reusable_sample_cache_output_path.exists():
                try:
                    os.link(reused_stage3_sample_cache_path, reusable_sample_cache_output_path)
                except OSError:
                    write_reusable_stage3_art_state_sample_cache(debug_stem, stage3_art_state_samples)
            elapsed_seconds = time.perf_counter() - stage_started_at
            stage_timings.append(
                build_stage_timing_entry(
                    'runtime_stage_1c_reusable_frame_payload',
                    'Runtime Stage 1C - Writing reusable Stage 2B frame payload',
                    elapsed_seconds,
                    len(stage3_art_state_samples),
                )
            )
            emit_stage_status(
                status_callback,
                f"Runtime Stage 1C - Linking reused Stage 2B frame payload into current debug folder complete in {format_stage_elapsed(elapsed_seconds)} ({len(stage3_art_state_samples)} sampled frames)",
            )
        else:
            emit_stage_status(status_callback, 'Runtime Stage 1C - Writing reusable Stage 2B frame payload started')
            stage_started_at = time.perf_counter()
            write_reusable_stage3_art_state_sample_cache(debug_stem, stage3_art_state_samples)
            elapsed_seconds = time.perf_counter() - stage_started_at
            stage_timings.append(
                build_stage_timing_entry(
                    'runtime_stage_1c_reusable_frame_payload',
                    'Runtime Stage 1C - Writing reusable Stage 2B frame payload',
                    elapsed_seconds,
                    len(stage3_art_state_samples),
                )
            )
            emit_stage_status(
                status_callback,
                f"Runtime Stage 1C - Writing reusable Stage 2B frame payload complete in {format_stage_elapsed(elapsed_seconds)} ({len(stage3_art_state_samples)} sampled frames)",
            )
    flush_debug_payload({
        'movement_evidence_records': serialized_records,
        'movement_spans': serialized_movement_spans,
        'candidate_unions': serialized_candidate_unions,
        'stage_timings': stage_timings,
    })

    emit_stage_status(status_callback, 'Runtime Stage 3A - Screening candidate unions started')
    stage_started_at = time.perf_counter()
    screened_candidate_unions = screen_stage3_candidate_unions(
        candidate_unions,
        records,
        sampled_frames=stage3_art_state_samples,
        chapter_range=chapter_range,
        settings=settings,
        status_callback=status_callback,
    )
    elapsed_seconds = time.perf_counter() - stage_started_at
    stage_timings.append(build_stage_timing_entry('runtime_stage_3a_union_screening', 'Runtime Stage 3A - Screening candidate unions', elapsed_seconds, len(screened_candidate_unions)))
    emit_stage_status(status_callback, f"Runtime Stage 3A - Screening candidate unions complete in {format_stage_elapsed(elapsed_seconds)} ({len(screened_candidate_unions)} screened unions)")
    serialized_screened_candidate_unions = [serialize_screened_candidate_union(screened_union) for screened_union in screened_candidate_unions]
    serialized_stage3_screening_traces = [serialize_stage3_screening_trace(screened_union) for screened_union in screened_candidate_unions]
    flush_debug_payload({
        'movement_evidence_records': serialized_records,
        'movement_spans': serialized_movement_spans,
        'candidate_unions': serialized_candidate_unions,
        'screened_candidate_unions': serialized_screened_candidate_unions,
        'stage3_screening_traces': serialized_stage3_screening_traces,
        'stage_timings': stage_timings,
    })

    emit_stage_status(status_callback, 'Runtime Stage 4 - Detecting probe-bands started')
    stage_started_at = time.perf_counter()
    classified_time_slices, serialized_stage4_subregion_debug, serialized_stage4_cell_reference_debug = classify_stage4_time_slices_with_subregion_debug(screened_candidate_unions, records, sampled_frames=stage3_art_state_samples, settings=settings)
    elapsed_seconds = time.perf_counter() - stage_started_at
    stage_timings.append(build_stage_timing_entry('runtime_stage_4_time_slice_classification', 'Runtime Stage 4 - Detecting probe-bands', elapsed_seconds, len(classified_time_slices)))
    emit_stage_status(status_callback, f"Runtime Stage 4 - Detecting probe-bands complete in {format_stage_elapsed(elapsed_seconds)} ({len(classified_time_slices)} probe-bands)")
    serialized_classified_time_slices = [serialize_classified_time_slice(time_slice) for time_slice in classified_time_slices]
    flush_debug_payload({
        'movement_evidence_records': serialized_records,
        'movement_spans': serialized_movement_spans,
        'candidate_unions': serialized_candidate_unions,
        'screened_candidate_unions': serialized_screened_candidate_unions,
        'stage3_screening_traces': serialized_stage3_screening_traces,
        'classified_time_slices': serialized_classified_time_slices,
        'stage4_subregion_debug': serialized_stage4_subregion_debug,
        'stage4_cell_reference_debug': serialized_stage4_cell_reference_debug,
        'stage_timings': stage_timings,
    })

    emit_stage_status(status_callback, 'Runtime Stage 5 - Running recursive refinement started')
    stage_started_at = time.perf_counter()
    refined_sub_slices = refine_stage5_sub_slices(screened_candidate_unions, classified_time_slices, records, sampled_frames=stage3_art_state_samples, settings=settings)
    elapsed_seconds = time.perf_counter() - stage_started_at
    stage_timings.append(build_stage_timing_entry('runtime_stage_5_recursive_refinement', 'Runtime Stage 5 - Running recursive refinement', elapsed_seconds, len(refined_sub_slices)))
    emit_stage_status(status_callback, f"Runtime Stage 5 - Running recursive refinement complete in {format_stage_elapsed(elapsed_seconds)} ({len(refined_sub_slices)} refined slices)")
    serialized_refined_sub_slices = [serialize_classified_time_slice(time_slice) for time_slice in refined_sub_slices]
    flush_debug_payload({
        'movement_evidence_records': serialized_records,
        'movement_spans': serialized_movement_spans,
        'candidate_unions': serialized_candidate_unions,
        'screened_candidate_unions': serialized_screened_candidate_unions,
        'stage3_screening_traces': serialized_stage3_screening_traces,
        'classified_time_slices': serialized_classified_time_slices,
        'stage4_subregion_debug': serialized_stage4_subregion_debug,
        'stage4_cell_reference_debug': serialized_stage4_cell_reference_debug,
        'refined_sub_slices': serialized_refined_sub_slices,
        'stage_timings': stage_timings,
    })

    emit_stage_status(status_callback, 'Runtime Stage 6 - Assembling final retained ranges started')
    stage_started_at = time.perf_counter()
    final_candidate_ranges = assemble_stage6_candidate_ranges(refined_sub_slices)
    elapsed_seconds = time.perf_counter() - stage_started_at
    stage_timings.append(build_stage_timing_entry('runtime_stage_6_final_ranges', 'Runtime Stage 6 - Assembling final retained ranges', elapsed_seconds, len(final_candidate_ranges)))
    emit_stage_status(status_callback, f"Runtime Stage 6 - Assembling final retained ranges complete in {format_stage_elapsed(elapsed_seconds)} ({len(final_candidate_ranges)} retained ranges)")
    serialized_final_candidate_ranges = [serialize_final_candidate_range(final_range) for final_range in final_candidate_ranges]

    total_elapsed_seconds = time.perf_counter() - total_started_at
    stage_timings.append(build_stage_timing_entry('stage_total', 'Total staged detector time', total_elapsed_seconds))
    emit_stage_status(status_callback, 'Staged detector timing summary:')
    for timing in stage_timings:
        stage_label = timing['stage_label']
        elapsed_hhmmss = timing['elapsed_hhmmss']
        item_count = timing.get('item_count')
        detail_suffix = '' if item_count is None else f' ({item_count} items)'
        emit_stage_status(status_callback, f"- {stage_label}: {elapsed_hhmmss}{detail_suffix}")

    debug_payload: dict[str, object] = {
        'movement_evidence_records': serialized_records,
        'movement_spans': serialized_movement_spans,
        'candidate_unions': serialized_candidate_unions,
        'screened_candidate_unions': serialized_screened_candidate_unions,
        'stage3_screening_traces': serialized_stage3_screening_traces,
        'classified_time_slices': serialized_classified_time_slices,
        'stage4_subregion_debug': serialized_stage4_subregion_debug,
        'stage4_cell_reference_debug': serialized_stage4_cell_reference_debug,
        'refined_sub_slices': serialized_refined_sub_slices,
        'final_candidate_ranges': serialized_final_candidate_ranges,
        'stage_timings': stage_timings,
    }
    flush_debug_payload(debug_payload)
    final_ranges = [(final_range.start_frame, final_range.end_frame) for final_range in final_candidate_ranges]
    return final_ranges, debug_payload
def build_staged_debug_output_path(debug_stem: Path, artifact_key: str, suffix: str) -> Path:
    artifact_title = DEBUG_SUMMARY_TITLE if artifact_key == 'summary' else DEBUG_ARTIFACT_TITLES.get(artifact_key, artifact_key)
    return debug_stem.with_name(f'{debug_stem.name} - {artifact_title}{suffix}')


def write_staged_debug_artifacts(debug_stem: Path, debug_payload: dict[str, object]) -> dict[str, Path]:
    debug_stem.parent.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}
    for section_name, items in debug_payload.items():
        output_path = build_staged_debug_output_path(debug_stem, section_name, '.json')
        output_path.write_text(json.dumps(items, indent=2), encoding='utf-8')
        output_paths[section_name] = output_path
        if section_name == 'movement_evidence_records' and isinstance(items, list):
            output_paths['movement_evidence_records_cache'] = write_precomputed_movement_evidence_record_cache(
                output_path,
                items,
            )
    reusable_sample_cache_path = build_staged_debug_output_path(debug_stem, 'reusable_stage3_art_state_samples', '.npz')
    if reusable_sample_cache_path.exists():
        output_paths['reusable_stage3_art_state_samples'] = reusable_sample_cache_path
    summary_output_path = build_staged_debug_output_path(debug_stem, 'summary', '.txt')
    summary_output_path.write_text(
        "\n".join(build_staged_debug_summary_lines(debug_payload)) + "\n",
        encoding='utf-8',
    )
    output_paths['summary'] = summary_output_path
    return output_paths


























































