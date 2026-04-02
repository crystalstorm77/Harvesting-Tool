from __future__ import annotations

# ============================================================
# SECTION A - Imports And Reused Detection Primitives
# ============================================================

from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

from harvesting_tool.detection import (
    ART_STATE_BOTTOM_RATIO,
    ART_STATE_LEFT_RATIO,
    ART_STATE_RIGHT_RATIO,
    ART_STATE_TOP_RATIO,
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
    extract_art_state_region,
    is_weak_art_change_signal,
    should_enter_active_state,
    should_remain_active_state,
)


# ============================================================
# SECTION B - Staged Detection Data Structures
# ============================================================

GridCoordinate = tuple[int, int]
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
STAGE3_MIN_REFERENCE_RECORDS = 2
STAGE3_SURVIVING_THRESHOLD = 0.40
STAGE3_MAX_REFERENCE_ACTIVITY = 0.15
STAGE3_MIN_CONTRAST_SCORE = 0.05
STAGE4_VALID_EVIDENCE_SCORE = 0.70
STAGE4_INVALID_EVIDENCE_SCORE = 0.30
STAGE4_MAX_REFERENCE_ACTIVITY = 0.18
STAGE4_MIN_CONTRAST_SCORE = 0.05
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
    stage3_mode: str = 'movement_records'
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

@dataclass(frozen=True)
class FinalCandidateRange:
    range_index: int
    parent_union_index: int
    start_frame: int
    end_frame: int
    start_time: str
    end_time: str
    source_classifications: tuple[str, ...]
    includes_retained_undetermined: bool
    retained_undetermined_count: int



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
) -> list[MovementEvidenceRecord]:
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('OpenCV is required for staged movement detection.') from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f'Unable to open video file: {video_path}')

    recent_persistent_masks: deque = deque(maxlen=TRAIL_MASK_WINDOW)
    sampled_frames: deque[dict[str, object]] = deque(maxlen=3)
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
                sampled_frames.append({'frame_index': current_frame, 'gray': extract_canvas_region(frame, cv2)})
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

    return records



def build_stage1_movement_spans(
    records: Iterable[MovementEvidenceRecord],
    settings: DetectorSettings,
) -> list[MovementSpan]:
    ordered_records = list(records)
    spans: list[MovementSpan] = []
    active_start_frame: int | None = None
    active_record_indices: list[int] = []
    active_footprint: set[GridCoordinate] = set()
    inactive_streak = 0
    recent_signals: list[tuple[int, bool]] = []
    recent_records: list[MovementEvidenceRecord] = []

    def close_span(end_frame: int, span_index: int) -> None:
        if active_start_frame is None or not active_record_indices:
            return
        spans.append(
            MovementSpan(
                span_index=span_index,
                start_frame=active_start_frame,
                end_frame=end_frame,
                start_time=Timecode(total_frames=active_start_frame).to_hhmmssff(),
                end_time=Timecode(total_frames=end_frame).to_hhmmssff(),
                footprint=frozenset(active_footprint),
                footprint_size=len(active_footprint),
                record_indices=tuple(active_record_indices),
            )
        )

    for record in ordered_records:
        prior_signals = recent_signals.copy()
        prior_records = recent_records.copy()
        opening_support = [recent_record for recent_record in prior_records if recent_record.movement_present]

        if active_start_frame is None:
            if record.movement_present and opening_support:
                backtracked_start = backtrack_event_start(prior_signals, record.frame_index)
                active_start_frame = backtracked_start
                active_record_indices = []
                active_footprint = set()
                for recent_record in prior_records:
                    if recent_record.frame_index < backtracked_start:
                        continue
                    if recent_record.movement_present:
                        active_record_indices.append(recent_record.record_index)
                        active_footprint.update(recent_record.touched_grid_coordinates)
                if record.record_index not in active_record_indices:
                    active_record_indices.append(record.record_index)
                active_footprint.update(record.touched_grid_coordinates)
                inactive_streak = 0
        else:
            if record.movement_present:
                inactive_streak = 0
                if record.record_index not in active_record_indices:
                    active_record_indices.append(record.record_index)
                active_footprint.update(record.touched_grid_coordinates)
            else:
                inactive_streak += 1
                if inactive_streak >= END_INACTIVE_SAMPLES:
                    close_span(record.frame_index, len(spans) + 1)
                    active_start_frame = None
                    active_record_indices = []
                    active_footprint = set()
                    inactive_streak = 0

        recent_signals.append((record.frame_index, bool(record.movement_present)))
        recent_signals[:] = recent_signals[-BACKTRACK_BUFFER_SAMPLES:]
        recent_records.append(record)
        recent_records[:] = recent_records[-BACKTRACK_BUFFER_SAMPLES:]

    if active_start_frame is not None and active_record_indices:
        close_span(ordered_records[-1].frame_index + settings.sample_stride, len(spans) + 1)

    minimum_frames = settings.min_burst_length.total_frames
    return [span for span in spans if (span.end_frame - span.start_frame) >= minimum_frames]

# ============================================================
# SECTION E - Stage 2 Candidate Union Construction
# ============================================================


MAX_STAGE2_EXPANSION_ABSOLUTE_CELLS = 6
MAX_STAGE2_EXPANSION_GROWTH_RATIO = 0.25



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



def merge_causes_large_stage2_expansion(
    current_union_footprint: frozenset[GridCoordinate],
    next_span_footprint: frozenset[GridCoordinate],
) -> bool:
    if not current_union_footprint:
        return False
    merged_union_footprint = current_union_footprint.union(next_span_footprint)
    new_cells_added = len(merged_union_footprint) - len(current_union_footprint)
    growth_ratio = new_cells_added / max(1, len(current_union_footprint))
    return (
        new_cells_added > MAX_STAGE2_EXPANSION_ABSOLUTE_CELLS
        and growth_ratio > MAX_STAGE2_EXPANSION_GROWTH_RATIO
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
        previous_span_support = span_has_spatial_support(next_span.footprint, previous_span.footprint)
        union_support = span_has_spatial_support(next_span.footprint, union_footprint)
        large_expansion = merge_causes_large_stage2_expansion(union_footprint, next_span.footprint)
        should_merge = False
        if temporal_gap <= STRONG_CONTINUITY_GAP_FRAMES and previous_span_support and not large_expansion:
            should_merge = True
        elif temporal_gap <= MAX_UNION_GAP_FRAMES and previous_span_support and union_support and not large_expansion:
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
) -> list[dict[str, object]]:
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
                        'canvas_gray': canvas_gray,
                        'art_gray': extract_art_state_region(canvas_gray),
                    }
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
        current_art_gray = window_samples[sample_index]['art_gray']
        next_art_gray = window_samples[sample_index + 1]['art_gray']
        instability_mask = build_art_state_change_mask(current_art_gray, next_art_gray, settings, cv2)
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
) -> tuple[dict[str, object] | None, int]:
    candidate_windows = build_stage3_reference_window_candidates(
        search_start,
        search_end,
        STAGE3_ART_STATE_BEFORE_WINDOW_FRAMES,
        settings.sample_stride,
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
        drift_mask = build_art_state_change_mask(post_baseline, sample['art_gray'], settings, cv2)
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



def crop_canvas_mask_to_art_state(mask):
    frame_height, frame_width = mask.shape[:2]
    left = int(frame_width * ART_STATE_LEFT_RATIO)
    right = int(frame_width * ART_STATE_RIGHT_RATIO)
    top = int(frame_height * ART_STATE_TOP_RATIO)
    bottom = int(frame_height * ART_STATE_BOTTOM_RATIO)
    return mask[top:bottom, left:right]



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
        persistent_mask = build_art_state_change_mask(pre_baseline, sample['art_gray'], settings, cv2)
        focused_persistent_mask = cv2.bitwise_and(persistent_mask, art_state_footprint_mask)
        persistence_scores.append(
            compute_stage3_art_state_persistent_difference_score(
                focused_persistent_mask,
                footprint_size,
                settings,
            )
        )
        instability_mask = build_art_state_change_mask(post_baseline, sample['art_gray'], settings, cv2)
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



def screen_candidate_union_with_art_state_prototype(
    candidate_union: CandidateUnion,
    records: Iterable[MovementEvidenceRecord],
    sampled_frames: list[dict[str, object]],
    chapter_range: ChapterRange,
    settings: DetectorSettings,
    previous_union_end: int | None,
    next_union_start: int | None,
    cv2,
) -> ScreenedCandidateUnion:
    ordered_records = sorted(records, key=lambda record: record.frame_index)
    within_union_records = collect_records_in_frame_range(
        ordered_records,
        candidate_union.start_frame,
        candidate_union.end_frame + 1,
    )
    (before_search_start, before_search_end), (after_search_start, after_search_end) = build_stage3_art_state_search_ranges(
        candidate_union,
        chapter_range,
    )
    selected_before_window, before_window_candidate_count = select_stage3_art_state_reference_window(
        search_start=before_search_start,
        search_end=before_search_end,
        union_anchor_frame=candidate_union.start_frame,
        records=ordered_records,
        sampled_frames=sampled_frames,
        settings=settings,
        cv2=cv2,
    )
    selected_after_window, after_window_candidate_count = select_stage3_art_state_reference_window(
        search_start=after_search_start,
        search_end=after_search_end,
        union_anchor_frame=candidate_union.end_frame,
        records=ordered_records,
        sampled_frames=sampled_frames,
        settings=settings,
        cv2=cv2,
    )
    before_window_start = selected_before_window["window_start"] if selected_before_window is not None else before_search_start
    before_window_end = selected_before_window["window_end"] if selected_before_window is not None else before_search_start
    after_window_start = selected_after_window["window_start"] if selected_after_window is not None else after_search_start
    after_window_end = selected_after_window["window_end"] if selected_after_window is not None else after_search_start
    reveal_search_start, reveal_search_end = build_stage3_reveal_search_range(
        after_window_end,
        chapter_range,
        next_union_start,
    )
    selected_reveal_window, reveal_window_candidate_count = select_stage3_art_state_reference_window(
        search_start=reveal_search_start,
        search_end=reveal_search_end,
        union_anchor_frame=after_window_end,
        records=ordered_records,
        sampled_frames=sampled_frames,
        settings=settings,
        cv2=cv2,
    )
    reveal_window_start = selected_reveal_window["window_start"] if selected_reveal_window is not None else reveal_search_start
    reveal_window_end = selected_reveal_window["window_end"] if selected_reveal_window is not None else reveal_search_start
    before_records = collect_records_in_frame_range(ordered_records, before_window_start, before_window_end)
    after_records = collect_records_in_frame_range(ordered_records, after_window_start, after_window_end)
    before_samples, pre_baseline = build_stage3_art_state_window_baseline(sampled_frames, before_window_start, before_window_end)
    after_samples, post_baseline = build_stage3_art_state_window_baseline(sampled_frames, after_window_start, after_window_end)
    reveal_samples, reveal_baseline = build_stage3_art_state_window_baseline(sampled_frames, reveal_window_start, reveal_window_end)

    mean_movement_strength = average_record_score(within_union_records, 'movement_strength_score')
    mean_temporal_persistence = average_record_score(within_union_records, 'temporal_persistence_score')
    mean_spatial_extent = average_record_score(within_union_records, 'spatial_extent_score')
    before_reference_activity = average_record_score(before_records, 'movement_strength_score')
    after_reference_activity = average_record_score(after_records, 'movement_strength_score')
    reference_windows_reliable = (
        selected_before_window is not None
        and selected_after_window is not None
        and len(before_samples) >= STAGE3_ART_STATE_MIN_SAMPLES
        and len(after_samples) >= STAGE3_ART_STATE_MIN_SAMPLES
        and pre_baseline is not None
        and post_baseline is not None
    )
    reveal_window_reliable = (
        selected_reveal_window is not None
        and len(reveal_samples) >= STAGE3_ART_STATE_MIN_SAMPLES
        and reveal_baseline is not None
    )

    persistent_difference_score = 0.0
    footprint_support_score = 0.0
    after_window_persistence_score = 0.0
    reveal_window_hold_score = 0.0
    if reference_windows_reliable:
        reference_canvas_shape = before_samples[0]['canvas_gray'].shape
        canvas_footprint_mask = build_union_canvas_footprint_mask(candidate_union, reference_canvas_shape)
        art_state_footprint_mask = crop_canvas_mask_to_art_state(canvas_footprint_mask)
        art_state_change_mask = build_art_state_change_mask(pre_baseline, post_baseline, settings, cv2)
        focused_change_mask = cv2.bitwise_and(art_state_change_mask, art_state_footprint_mask)
        persistent_difference_score = compute_stage3_art_state_persistent_difference_score(
            focused_change_mask,
            candidate_union.union_footprint_size,
            settings,
        )
        footprint_support_score = compute_stage3_art_state_footprint_support_score(
            focused_change_mask,
            candidate_union.union_footprint_size,
        )
        after_window_persistence_score = compute_stage3_art_state_after_window_persistence_score(
            pre_baseline,
            post_baseline,
            after_samples,
            art_state_footprint_mask,
            candidate_union.union_footprint_size,
            settings,
            cv2,
        )
        if reveal_window_reliable:
            reveal_window_hold_score = compute_stage3_art_state_reveal_hold_score(
                pre_baseline,
                post_baseline,
                reveal_baseline,
                reveal_samples,
                art_state_footprint_mask,
                candidate_union.union_footprint_size,
                settings,
                cv2,
            )

    lasting_change_evidence_score = compute_stage3_art_state_evidence_score(
        persistent_difference_score,
        footprint_support_score,
        after_window_persistence_score,
        reveal_window_hold_score if reveal_window_reliable else STAGE3_ART_STATE_MISSING_REVEAL_SCORE,
    )
    reference_activity_ceiling = max(before_reference_activity, after_reference_activity)
    contrast_score = lasting_change_evidence_score - reference_activity_ceiling
    has_meaningful_union_activity = (
        bool(within_union_records)
        and candidate_union.union_footprint_size > 0
        and lasting_change_evidence_score >= STAGE3_SURVIVING_THRESHOLD
    )
    has_meaningful_footprint_support = (
        footprint_support_score >= STAGE3_ART_STATE_MIN_FOOTPRINT_SUPPORT_SCORE
    )
    fallback_post_reference_without_reveal = (
        not reveal_window_reliable
        and selected_after_window is not None
        and str(selected_after_window['tier']) == 'fallback'
    )
    fallback_post_reference_supports_survival = (
        candidate_union.union_footprint_size >= STAGE3_ART_STATE_FALLBACK_REFERENCE_MIN_FOOTPRINT
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
    elif not has_meaningful_footprint_support:
        screening_result = 'rejected'
        surviving = False
        provisional_survival = False
        reason = 'insufficient_footprint_support'
    elif fallback_post_reference_without_reveal and not fallback_post_reference_supports_survival:
        screening_result = 'rejected'
        surviving = False
        provisional_survival = False
        reason = 'fallback_post_reference_too_small'
    elif (
        reference_activity_ceiling <= STAGE3_MAX_REFERENCE_ACTIVITY
        or contrast_score >= STAGE3_MIN_CONTRAST_SCORE
    ):
        screening_result = 'surviving'
        surviving = True
        provisional_survival = False
        reason = 'art_state_change_supported'
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
        stage3_mode='art_state_prototype',
        stage3_alignment_mode='none',
        stage3_persistent_difference_score=persistent_difference_score,
        stage3_footprint_support_score=footprint_support_score,
        stage3_after_window_persistence_score=after_window_persistence_score,
        stage3_before_window_start=Timecode(total_frames=before_window_start).to_hhmmssff(),
        stage3_before_window_end=Timecode(total_frames=before_window_end).to_hhmmssff(),
        stage3_after_window_start=Timecode(total_frames=after_window_start).to_hhmmssff(),
        stage3_after_window_end=Timecode(total_frames=after_window_end).to_hhmmssff(),
        stage3_reveal_window_start=Timecode(total_frames=reveal_window_start).to_hhmmssff(),
        stage3_reveal_window_end=Timecode(total_frames=reveal_window_end).to_hhmmssff(),
        stage3_before_sample_count=len(before_samples),
        stage3_after_sample_count=len(after_samples),
        stage3_reveal_sample_count=len(reveal_samples),
        stage3_before_window_quality_score=0.0 if selected_before_window is None else float(selected_before_window['quality_score']),
        stage3_after_window_quality_score=0.0 if selected_after_window is None else float(selected_after_window['quality_score']),
        stage3_reveal_window_quality_score=0.0 if selected_reveal_window is None else float(selected_reveal_window['quality_score']),
        stage3_before_window_candidate_count=before_window_candidate_count,
        stage3_after_window_candidate_count=after_window_candidate_count,
        stage3_reveal_window_candidate_count=reveal_window_candidate_count,
        stage3_before_window_tier=None if selected_before_window is None else str(selected_before_window['tier']),
        stage3_after_window_tier=None if selected_after_window is None else str(selected_after_window['tier']),
        stage3_reveal_window_tier=None if selected_reveal_window is None else str(selected_reveal_window['tier']),
        stage3_reveal_window_hold_score=reveal_window_hold_score,
    )


def screen_stage3_candidate_unions(
    candidate_unions: Iterable[CandidateUnion],
    records: Iterable[MovementEvidenceRecord],
    sampled_frames: list[dict[str, object]] | None = None,
    chapter_range: ChapterRange | None = None,
    settings: DetectorSettings | None = None,
    use_art_state_prototype: bool = False,
) -> list[ScreenedCandidateUnion]:
    ordered_candidate_unions = list(candidate_unions)
    if not use_art_state_prototype:
        return [screen_candidate_union(candidate_union, records) for candidate_union in ordered_candidate_unions]

    if sampled_frames is None or chapter_range is None or settings is None:
        raise ValueError('Stage 3 art-state prototype requires sampled_frames, chapter_range, and settings.')

    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('OpenCV is required for staged Stage 3 art-state screening.') from exc

    screened_unions: list[ScreenedCandidateUnion] = []
    for candidate_union_index, candidate_union in enumerate(ordered_candidate_unions):
        previous_union_end = None if candidate_union_index == 0 else ordered_candidate_unions[candidate_union_index - 1].end_frame
        next_union_start = None if candidate_union_index == (len(ordered_candidate_unions) - 1) else ordered_candidate_unions[candidate_union_index + 1].start_frame
        screened_unions.append(
            screen_candidate_union_with_art_state_prototype(
                candidate_union=candidate_union,
                records=records,
                sampled_frames=sampled_frames,
                chapter_range=chapter_range,
                settings=settings,
                previous_union_end=previous_union_end,
                next_union_start=next_union_start,
                cv2=cv2,
            )
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

    midpoint = union_start + ((union_end - union_start) // 2)
    if midpoint <= union_start or midpoint >= union_end:
        return [(union_start, union_end)]
    return [(union_start, midpoint), (midpoint, union_end)]



def classify_time_slice(
    screened_candidate_union: ScreenedCandidateUnion,
    slice_index: int,
    slice_level: int,
    slice_start: int,
    slice_end: int,
    records: Iterable[MovementEvidenceRecord],
    reference_window_frames: int = STAGE3_REFERENCE_WINDOW_FRAMES,
) -> ClassifiedTimeSlice:
    ordered_records = sorted(records, key=lambda record: record.frame_index)
    slice_records = collect_records_in_frame_range(ordered_records, slice_start, slice_end)
    before_records = collect_records_in_frame_range(
        ordered_records,
        max(0, slice_start - reference_window_frames),
        slice_start,
    )
    after_records = collect_records_in_frame_range(
        ordered_records,
        slice_end,
        slice_end + reference_window_frames,
    )
    footprint = build_footprint_from_records(slice_records)
    footprint_size = len(footprint)
    before_reference_activity = average_record_score(before_records, 'movement_strength_score')
    after_reference_activity = average_record_score(after_records, 'movement_strength_score')
    lasting_change_evidence_score = compute_slice_lasting_change_evidence_score(
        slice_records=slice_records,
        footprint_size=footprint_size,
        after_reference_activity=after_reference_activity,
    )
    reference_windows_reliable = (
        len(before_records) >= STAGE3_MIN_REFERENCE_RECORDS
        and len(after_records) >= STAGE3_MIN_REFERENCE_RECORDS
    )
    reference_activity_ceiling = max(before_reference_activity, after_reference_activity)
    reference_activity_floor = min(before_reference_activity, after_reference_activity)
    contrast_score = lasting_change_evidence_score - reference_activity_ceiling
    quiet_reference_available = reference_activity_floor <= STAGE4_MAX_REFERENCE_ACTIVITY
    both_references_active = (
        before_reference_activity > STAGE4_MAX_REFERENCE_ACTIVITY
        and after_reference_activity > STAGE4_MAX_REFERENCE_ACTIVITY
    )
    relaxed_boundary_contrast_score = STAGE4_MIN_CONTRAST_SCORE * 0.8
    long_strong_union_supports_refinement = (
        (screened_candidate_union.candidate_union.end_frame - screened_candidate_union.candidate_union.start_frame)
        >= LONG_STRONG_UNION_MIN_FRAMES
        and screened_candidate_union.lasting_change_evidence_score >= LONG_STRONG_UNION_STAGE3_SCORE_FLOOR
    )
    if not slice_records or footprint_size == 0 or lasting_change_evidence_score <= STAGE4_INVALID_EVIDENCE_SCORE:
        classification = 'invalid'
        reason = 'weak_slice_activity'
    elif not reference_windows_reliable:
        classification = 'undetermined'
        reason = 'reference_windows_unreliable'
    elif (
        lasting_change_evidence_score >= STAGE4_VALID_EVIDENCE_SCORE
        and (
            (contrast_score >= STAGE4_MIN_CONTRAST_SCORE
             and (
                 not both_references_active
                 or screened_candidate_union.lasting_change_evidence_score >= LONG_STRONG_UNION_STAGE3_SCORE_FLOOR
             ))
            or (quiet_reference_available and contrast_score >= relaxed_boundary_contrast_score)
            or (quiet_reference_available and reference_activity_ceiling <= STAGE4_MAX_REFERENCE_ACTIVITY)
        )
    ):
        classification = 'valid'
        reason = 'slice_activity_supported'
    elif quiet_reference_available and reference_activity_ceiling > STAGE4_MAX_REFERENCE_ACTIVITY:
        classification = 'undetermined'
        reason = 'mixed_reference_activity'
    elif contrast_score < 0 and both_references_active:
        if (
            lasting_change_evidence_score >= STRONG_ACTIVE_REFERENCE_UNDETERMINED_FLOOR
            or (
                long_strong_union_supports_refinement
                and lasting_change_evidence_score >= LONG_STRONG_UNION_ACTIVE_REFERENCE_UNDETERMINED_FLOOR
            )
        ):
            classification = 'undetermined'
            reason = 'reference_windows_too_active'
        else:
            classification = 'invalid'
            reason = 'reference_windows_too_active'
    else:
        classification = 'undetermined'
        reason = 'mixed_slice_evidence'

    return ClassifiedTimeSlice(
        slice_index=slice_index,
        parent_union_index=screened_candidate_union.candidate_union.union_index,
        slice_level=slice_level,
        start_frame=slice_start,
        end_frame=slice_end,
        start_time=Timecode(total_frames=slice_start).to_hhmmssff(),
        end_time=Timecode(total_frames=slice_end).to_hhmmssff(),
        footprint=footprint,
        footprint_size=footprint_size,
        within_slice_record_count=len(slice_records),
        classification=classification,
        reason=reason,
        lasting_change_evidence_score=lasting_change_evidence_score,
        before_reference_activity=before_reference_activity,
        after_reference_activity=after_reference_activity,
        reference_windows_reliable=reference_windows_reliable,
    )



def classify_stage4_time_slices(
    screened_candidate_unions: Iterable[ScreenedCandidateUnion],
    records: Iterable[MovementEvidenceRecord],
) -> list[ClassifiedTimeSlice]:
    ordered_records = sorted(records, key=lambda record: record.frame_index)
    classified_slices: list[ClassifiedTimeSlice] = []

    for screened_candidate_union in screened_candidate_unions:
        if not screened_candidate_union.surviving:
            continue
        slice_ranges = build_stage4_time_slice_ranges(screened_candidate_union)
        for slice_index, (slice_start, slice_end) in enumerate(slice_ranges, start=1):
            classified_slices.append(
                classify_time_slice(
                    screened_candidate_union=screened_candidate_union,
                    slice_index=slice_index,
                    slice_level=0,
                    slice_start=slice_start,
                    slice_end=slice_end,
                    records=ordered_records,
                )
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
) -> ClassifiedTimeSlice:
    if (
        allow_valid_anchor_promotion
        and time_slice.classification == 'undetermined'
        and time_slice.footprint_size >= STAGE5_VALID_ANCHOR_PROMOTION_MIN_FOOTPRINT
        and time_slice.lasting_change_evidence_score >= STAGE5_VALID_ANCHOR_PROMOTION_MIN_EVIDENCE
    ):
        return replace(time_slice, classification='valid', reason='slice_activity_supported')
    if (
        allow_reference_unreliable_rescue
        and time_slice.classification == 'undetermined'
        and time_slice.footprint_size > 0
        and time_slice.lasting_change_evidence_score >= REFERENCE_UNRELIABLE_RESCUE_FLOOR
    ):
        return replace(time_slice, classification='boundary', reason='reference_unreliable_minimum_size_reached')
    if (
        time_slice.classification == 'undetermined'
        and time_slice.footprint_size > 0
        and time_slice.lasting_change_evidence_score >= ROCKY_MINIMUM_SIZE_EVIDENCE_FLOOR
    ):
        return replace(time_slice, classification='boundary', reason='minimum_subdivision_size_reached')
    if (
        allow_long_strong_union_rocky_rescue
        and time_slice.classification == 'undetermined'
        and time_slice.footprint_size > 0
        and time_slice.lasting_change_evidence_score >= LONG_STRONG_UNION_ACTIVE_REFERENCE_UNDETERMINED_FLOOR
    ):
        return replace(time_slice, classification='boundary', reason='long_strong_union_minimum_size_reached')
    if (
        allow_structural_gap_rescue
        and time_slice.classification == 'invalid'
        and time_slice.reason == 'reference_windows_too_active'
        and time_slice.footprint_size > 0
        and time_slice.lasting_change_evidence_score >= STRUCTURAL_GAP_RESCUE_FLOOR
    ):
        return replace(time_slice, classification='boundary', reason='structural_gap_minimum_size_reached')
    if (
        allow_art_state_supported_rescue
        and time_slice.classification == 'invalid'
        and time_slice.reason == 'reference_windows_too_active'
        and time_slice.footprint_size >= ART_STATE_SUPPORTED_MINIMUM_RESCUE_FOOTPRINT
        and time_slice.lasting_change_evidence_score >= ART_STATE_SUPPORTED_RESCUE_FLOOR
    ):
        return replace(time_slice, classification='boundary', reason='art_state_supported_minimum_size_reached')
    if (
        allow_high_parent_activity_rescue
        and time_slice.classification == 'invalid'
        and time_slice.reason == 'reference_windows_too_active'
        and time_slice.footprint_size > 0
        and time_slice.lasting_change_evidence_score >= HIGH_PARENT_ACTIVITY_RESCUE_FLOOR
    ):
        return replace(time_slice, classification='boundary', reason='high_parent_activity_minimum_size_reached')
    if (
        allow_active_reference_rescue
        and time_slice.classification == 'invalid'
        and time_slice.reason == 'reference_windows_too_active'
        and time_slice.footprint_size > 0
        and time_slice.lasting_change_evidence_score >= ACTIVE_REFERENCE_ROCKY_RESCUE_FLOOR
    ):
        return replace(time_slice, classification='boundary', reason='active_reference_minimum_size_reached')
    return replace(time_slice, reason='minimum_subdivision_size_reached')


def refine_stage5_sub_slices(
    screened_candidate_unions: Iterable[ScreenedCandidateUnion],
    classified_time_slices: Iterable[ClassifiedTimeSlice],
    records: Iterable[MovementEvidenceRecord],
    minimum_subdivision_frames: int = STAGE5_MIN_SUBDIVISION_FRAMES,
) -> list[ClassifiedTimeSlice]:
    union_lookup = {
        screened_candidate_union.candidate_union.union_index: screened_candidate_union
        for screened_candidate_union in screened_candidate_unions
    }
    ordered_records = sorted(records, key=lambda record: record.frame_index)
    unions_with_valid_slice = {
        time_slice.parent_union_index
        for time_slice in classified_time_slices
        if time_slice.classification == 'valid'
    }
    long_strong_unions = {
        screened_candidate_union.candidate_union.union_index
        for screened_candidate_union in screened_candidate_unions
        if (
            not screened_candidate_union.provisional_survival
            and (screened_candidate_union.candidate_union.end_frame - screened_candidate_union.candidate_union.start_frame)
            >= LONG_STRONG_UNION_MIN_FRAMES
            and screened_candidate_union.lasting_change_evidence_score >= LONG_STRONG_UNION_STAGE3_SCORE_FLOOR
        )
    }
    art_state_supported_unions = {
        screened_candidate_union.candidate_union.union_index
        for screened_candidate_union in screened_candidate_unions
        if (
            screened_candidate_union.stage3_mode == 'art_state_prototype'
            and screened_candidate_union.surviving
            and not screened_candidate_union.provisional_survival
            and screened_candidate_union.lasting_change_evidence_score >= ART_STATE_SUPPORTED_UNION_STAGE3_SCORE_FLOOR
            and screened_candidate_union.stage3_footprint_support_score >= ART_STATE_SUPPORTED_UNION_FOOTPRINT_SUPPORT_FLOOR
            and screened_candidate_union.stage3_footprint_support_score < ART_STATE_SUPPORTED_UNION_MAX_FOOTPRINT_SUPPORT
        )
    }
    def refine_slice(
        time_slice: ClassifiedTimeSlice,
        allow_high_parent_activity_rescue: bool = False,
        allow_reference_unreliable_rescue: bool = False,
        allow_structural_gap_rescue: bool = False,
        allow_valid_anchor_promotion: bool = False,
    ) -> list[ClassifiedTimeSlice]:
        allow_active_reference_rescue = time_slice.parent_union_index in unions_with_valid_slice
        allow_long_strong_union_rocky_rescue = time_slice.parent_union_index in long_strong_unions
        allow_art_state_supported_rescue = time_slice.parent_union_index in art_state_supported_unions
        current_slice_supports_high_parent_activity_rescue = (
            time_slice.classification == 'undetermined'
            and time_slice.reason in {'mixed_reference_activity', 'reference_windows_too_active'}
            and time_slice.lasting_change_evidence_score >= HIGH_PARENT_ACTIVITY_PARENT_SCORE_FLOOR
        )
        current_slice_supports_reference_unreliable_rescue = (
            time_slice.classification == 'undetermined'
            and time_slice.reason == 'reference_windows_unreliable'
            and time_slice.lasting_change_evidence_score >= REFERENCE_UNRELIABLE_PARENT_SCORE_FLOOR
        )
        current_slice_supports_structural_gap_rescue = (
            time_slice.classification == 'undetermined'
            and time_slice.lasting_change_evidence_score >= STRUCTURAL_GAP_PARENT_SCORE_FLOOR
            and (
                (time_slice.reason == 'reference_windows_too_active'
                 and time_slice.parent_union_index in long_strong_unions)
                or (
                    time_slice.reason == 'mixed_reference_activity'
                    and time_slice.before_reference_activity <= STAGE4_MAX_REFERENCE_ACTIVITY
                    and time_slice.after_reference_activity > STAGE4_MAX_REFERENCE_ACTIVITY
                )
            )
        )
        slice_duration = time_slice.end_frame - time_slice.start_frame

        if time_slice.classification == 'invalid':
            if (
                (allow_active_reference_rescue or allow_high_parent_activity_rescue or allow_structural_gap_rescue or allow_art_state_supported_rescue)
                and time_slice.reason == 'reference_windows_too_active'
                and slice_duration <= minimum_subdivision_frames
            ):
                return [
                    classify_stage5_minimum_size_leaf(
                        time_slice,
                        allow_active_reference_rescue=allow_active_reference_rescue,
                        allow_long_strong_union_rocky_rescue=allow_long_strong_union_rocky_rescue,
                        allow_high_parent_activity_rescue=allow_high_parent_activity_rescue,
                        allow_reference_unreliable_rescue=allow_reference_unreliable_rescue,
                        allow_structural_gap_rescue=allow_structural_gap_rescue,
                        allow_art_state_supported_rescue=allow_art_state_supported_rescue,
                        allow_valid_anchor_promotion=allow_valid_anchor_promotion,
                    )
                ]
            if not (
                (allow_active_reference_rescue or allow_high_parent_activity_rescue or allow_structural_gap_rescue or allow_art_state_supported_rescue)
                and time_slice.reason == 'reference_windows_too_active'
                and slice_duration > minimum_subdivision_frames
            ):
                return [time_slice]

        elif time_slice.classification != 'undetermined':
            return [time_slice]

        if slice_duration <= 1:
            return [
                classify_stage5_minimum_size_leaf(
                    time_slice,
                    allow_active_reference_rescue=allow_active_reference_rescue,
                    allow_long_strong_union_rocky_rescue=allow_long_strong_union_rocky_rescue,
                    allow_high_parent_activity_rescue=allow_high_parent_activity_rescue,
                    allow_reference_unreliable_rescue=allow_reference_unreliable_rescue,
                    allow_structural_gap_rescue=allow_structural_gap_rescue,
                    allow_art_state_supported_rescue=allow_art_state_supported_rescue,
                    allow_valid_anchor_promotion=allow_valid_anchor_promotion,
                )
            ]

        if slice_duration <= minimum_subdivision_frames:
            return [
                classify_stage5_minimum_size_leaf(
                    time_slice,
                    allow_active_reference_rescue=allow_active_reference_rescue,
                    allow_long_strong_union_rocky_rescue=allow_long_strong_union_rocky_rescue,
                    allow_high_parent_activity_rescue=allow_high_parent_activity_rescue,
                    allow_reference_unreliable_rescue=allow_reference_unreliable_rescue,
                    allow_structural_gap_rescue=allow_structural_gap_rescue,
                    allow_art_state_supported_rescue=allow_art_state_supported_rescue,
                    allow_valid_anchor_promotion=allow_valid_anchor_promotion,
                )
            ]

        screened_candidate_union = union_lookup.get(time_slice.parent_union_index)
        if screened_candidate_union is None:
            return [replace(time_slice, reason='missing_parent_union')]

        sub_slice_ranges = build_stage5_sub_slice_ranges(time_slice)
        if len(sub_slice_ranges) <= 1:
            return [
                classify_stage5_minimum_size_leaf(
                    time_slice,
                    allow_active_reference_rescue=time_slice.parent_union_index in unions_with_valid_slice,
                    allow_long_strong_union_rocky_rescue=time_slice.parent_union_index in long_strong_unions,
                    allow_high_parent_activity_rescue=allow_high_parent_activity_rescue,
                    allow_reference_unreliable_rescue=allow_reference_unreliable_rescue,
                    allow_structural_gap_rescue=allow_structural_gap_rescue,
                    allow_art_state_supported_rescue=allow_art_state_supported_rescue,
                    allow_valid_anchor_promotion=allow_valid_anchor_promotion,
                )
            ]

        classified_sub_slices: list[ClassifiedTimeSlice] = []
        for slice_index, (slice_start, slice_end) in enumerate(sub_slice_ranges, start=1):
            sub_slice_duration = slice_end - slice_start
            sub_slice_reference_window = max(
                minimum_subdivision_frames,
                min(STAGE3_REFERENCE_WINDOW_FRAMES, sub_slice_duration),
            )
            classified_sub_slices.append(
                replace(
                    classify_time_slice(
                        screened_candidate_union,
                        slice_index=slice_index,
                        slice_level=time_slice.slice_level + 1,
                        slice_start=slice_start,
                        slice_end=slice_end,
                        records=ordered_records,
                        reference_window_frames=sub_slice_reference_window,
                    ),
                    parent_range=(time_slice.start_frame, time_slice.end_frame),
                )
            )

        refined_leaves: list[ClassifiedTimeSlice] = []
        for classified_sub_slice in classified_sub_slices:
            coherent_sibling_support = any(
                sibling_slice is not classified_sub_slice
                and sibling_slice.classification in {'valid', 'undetermined'}
                and sibling_slice.lasting_change_evidence_score >= STAGE5_VALID_ANCHOR_PROMOTION_MIN_SIBLING_EVIDENCE
                and compute_stage6_footprint_overlap(classified_sub_slice, sibling_slice) >= STAGE5_VALID_ANCHOR_PROMOTION_MIN_SIBLING_OVERLAP
                for sibling_slice in classified_sub_slices
            )
            local_valid_anchor_promotion = (
                allow_valid_anchor_promotion
                or (
                    coherent_sibling_support
                    and classified_sub_slice.classification == 'undetermined'
                    and classified_sub_slice.reason in {'mixed_reference_activity', 'reference_windows_too_active', 'reference_windows_unreliable', 'mixed_slice_evidence'}
                    and classified_sub_slice.footprint_size >= STAGE5_VALID_ANCHOR_PROMOTION_MIN_FOOTPRINT
                    and classified_sub_slice.lasting_change_evidence_score >= STAGE5_VALID_ANCHOR_PROMOTION_MIN_EVIDENCE
                    and (
                        classified_sub_slice.parent_union_index in long_strong_unions
                        or classified_sub_slice.parent_union_index in art_state_supported_unions
                    )
                )
            )
            refined_leaves.extend(
                refine_slice(
                    classified_sub_slice,
                    allow_high_parent_activity_rescue=(
                        allow_high_parent_activity_rescue
                        or current_slice_supports_high_parent_activity_rescue
                    ),
                    allow_reference_unreliable_rescue=(
                        allow_reference_unreliable_rescue
                        or current_slice_supports_reference_unreliable_rescue
                    ),
                    allow_structural_gap_rescue=(
                        allow_structural_gap_rescue
                        or current_slice_supports_structural_gap_rescue
                    ),
                    allow_valid_anchor_promotion=local_valid_anchor_promotion,
                )
            )
        return refined_leaves

    refined_slices: list[ClassifiedTimeSlice] = []
    for time_slice in classified_time_slices:
        refined_slices.extend(refine_slice(time_slice))
    return refined_slices


def is_stage6_candidate_slice(
    time_slice: ClassifiedTimeSlice,
    maximum_retained_frames: int = STAGE5_MIN_SUBDIVISION_FRAMES,
) -> bool:
    if time_slice.classification == 'valid':
        return True
    if time_slice.classification != 'boundary':
        return False
    if time_slice.reason not in {'minimum_subdivision_size_reached', 'active_reference_minimum_size_reached', 'long_strong_union_minimum_size_reached', 'high_parent_activity_minimum_size_reached', 'reference_unreliable_minimum_size_reached', 'structural_gap_minimum_size_reached', 'art_state_supported_minimum_size_reached'}:
        return False
    return (time_slice.end_frame - time_slice.start_frame) <= maximum_retained_frames



def supports_extended_stage6_gap(slice_info: ClassifiedTimeSlice) -> bool:
    return slice_info.reason in {
        'long_strong_union_minimum_size_reached',
        'structural_gap_minimum_size_reached',
        'reference_unreliable_minimum_size_reached',
    }


def compute_stage6_footprint_overlap(first_slice: ClassifiedTimeSlice, second_slice: ClassifiedTimeSlice) -> float:
    if not first_slice.footprint or not second_slice.footprint:
        return 0.0
    shared_footprint = len(first_slice.footprint & second_slice.footprint)
    combined_footprint = len(first_slice.footprint | second_slice.footprint)
    if combined_footprint == 0:
        return 0.0
    return shared_footprint / combined_footprint


def build_stage6_candidate_groups(
    refined_slices: Iterable[ClassifiedTimeSlice],
    maximum_retained_frames: int = STAGE5_MIN_SUBDIVISION_FRAMES,
    merge_gap_frames: int = STAGE5_MIN_SUBDIVISION_FRAMES,
) -> list[list[ClassifiedTimeSlice]]:
    ordered_slices = [
        slice_info
        for slice_info in sorted(
            refined_slices,
            key=lambda slice_info: (slice_info.parent_union_index, slice_info.start_frame, slice_info.end_frame),
        )
        if is_stage6_candidate_slice(slice_info, maximum_retained_frames=maximum_retained_frames)
    ]
    if not ordered_slices:
        return []

    candidate_groups: list[list[ClassifiedTimeSlice]] = []
    current_group: list[ClassifiedTimeSlice] = [ordered_slices[0]]
    current_group_supports_extended_gap = supports_extended_stage6_gap(ordered_slices[0])
    current_group_extended_gap_frames = 0
    for next_slice in ordered_slices[1:]:
        previous_slice = current_group[-1]
        same_union = next_slice.parent_union_index == previous_slice.parent_union_index
        gap_frames = next_slice.start_frame - previous_slice.end_frame
        close_enough = gap_frames <= merge_gap_frames
        footprint_overlap = compute_stage6_footprint_overlap(previous_slice, next_slice)
        low_overlap_valid_boundary = (
            gap_frames > 0
            and ('valid' in {previous_slice.classification, next_slice.classification})
            and footprint_overlap < STAGE6_VALID_MERGE_MIN_FOOTPRINT_OVERLAP
        )
        next_slice_supports_extended_gap = supports_extended_stage6_gap(next_slice)
        extended_gap_allowed = (
            same_union
            and not close_enough
            and gap_frames <= STAGE6_SAME_UNION_MERGE_GAP_FRAMES
            and (current_group_supports_extended_gap or next_slice_supports_extended_gap)
            and (current_group_extended_gap_frames + gap_frames) <= STAGE6_EXTENDED_GAP_BUDGET_FRAMES
        )
        if same_union and not low_overlap_valid_boundary and (close_enough or extended_gap_allowed):
            current_group.append(next_slice)
            current_group_supports_extended_gap = (
                current_group_supports_extended_gap or next_slice_supports_extended_gap
            )
            if extended_gap_allowed:
                current_group_extended_gap_frames += gap_frames
        else:
            candidate_groups.append(current_group)
            current_group = [next_slice]
            current_group_supports_extended_gap = next_slice_supports_extended_gap
            current_group_extended_gap_frames = 0
    candidate_groups.append(current_group)
    return candidate_groups

def should_keep_stage6_group(candidate_group: list[ClassifiedTimeSlice]) -> bool:
    if not candidate_group:
        return False
    if any(slice_info.classification == 'valid' for slice_info in candidate_group):
        return True

    boundary_slice_count = len(candidate_group)
    average_boundary_evidence = sum(slice_info.lasting_change_evidence_score for slice_info in candidate_group) / boundary_slice_count
    contains_abrupt_canvas_change = any(
        slice_info.footprint_size >= STAGE6_ABRUPT_CANVAS_CHANGE_MIN_FOOTPRINT
        for slice_info in candidate_group
    )
    total_duration = candidate_group[-1].end_frame - candidate_group[0].start_frame
    abrupt_canvas_change_survival = (
        contains_abrupt_canvas_change
        and total_duration <= STAGE6_ABRUPT_CANVAS_CHANGE_MAX_FRAMES
        and boundary_slice_count <= STAGE6_ABRUPT_CANVAS_CHANGE_MAX_SLICE_COUNT
        and average_boundary_evidence >= STAGE6_ABRUPT_CANVAS_CHANGE_MIN_EVIDENCE
    )
    return abrupt_canvas_change_survival

def assemble_stage6_candidate_ranges(
    refined_slices: Iterable[ClassifiedTimeSlice],
    retained_undetermined_max_frames: int = STAGE5_MIN_SUBDIVISION_FRAMES,
    merge_gap_frames: int = STAGE5_MIN_SUBDIVISION_FRAMES,
) -> list[FinalCandidateRange]:
    candidate_groups = build_stage6_candidate_groups(
        refined_slices,
        maximum_retained_frames=retained_undetermined_max_frames,
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
        retained_undetermined_count = sum(1 for slice_info in candidate_group if slice_info.classification == 'boundary')
        final_ranges.append(
            FinalCandidateRange(
                range_index=range_index,
                parent_union_index=candidate_group[0].parent_union_index,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=Timecode(total_frames=start_frame).to_hhmmssff(),
                end_time=Timecode(total_frames=end_frame).to_hhmmssff(),
                source_classifications=source_classifications,
                includes_retained_undetermined=retained_undetermined_count > 0,
                retained_undetermined_count=retained_undetermined_count,
            )
        )
    return final_ranges
# ============================================================
# SECTION J - Staged Pipeline Execution And Debug Output
# ============================================================

import json


def serialize_movement_evidence_record(record: MovementEvidenceRecord) -> dict[str, object]:
    return {
        'record_index': record.record_index,
        'evaluation_point_timecode': record.evaluation_point_timecode,
        'frame_index': record.frame_index,
        'movement_present': record.movement_present,
        'touched_grid_coordinates': [list(coordinate) for coordinate in record.touched_grid_coordinates],
        'touched_grid_coordinate_count': record.touched_grid_coordinate_count,
        'change_magnitude_score': round(record.change_magnitude_score, 6),
        'spatial_extent_score': round(record.spatial_extent_score, 6),
        'temporal_persistence_score': round(record.temporal_persistence_score, 6),
        'movement_strength_score': round(record.movement_strength_score, 6),
        'opening_signal': record.opening_signal,
        'continuation_signal': record.continuation_signal,
        'weak_signal': record.weak_signal,
    }



def serialize_movement_span(span: MovementSpan) -> dict[str, object]:
    return {
        'span_index': span.span_index,
        'start_frame': span.start_frame,
        'end_frame': span.end_frame,
        'start_time': span.start_time,
        'end_time': span.end_time,
        'footprint': [list(coordinate) for coordinate in sorted(span.footprint)],
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
        'footprint_size': time_slice.footprint_size,
        'within_slice_record_count': time_slice.within_slice_record_count,
        'classification': time_slice.classification,
        'reason': time_slice.reason,
        'lasting_change_evidence_score': round(time_slice.lasting_change_evidence_score, 6),
        'before_reference_activity': round(time_slice.before_reference_activity, 6),
        'after_reference_activity': round(time_slice.after_reference_activity, 6),
        'reference_windows_reliable': time_slice.reference_windows_reliable,
        'parent_range': list(time_slice.parent_range) if time_slice.parent_range is not None else None,
    }



def serialize_final_candidate_range(final_range: FinalCandidateRange) -> dict[str, object]:
    return {
        'range_index': final_range.range_index,
        'parent_union_index': final_range.parent_union_index,
        'start_frame': final_range.start_frame,
        'end_frame': final_range.end_frame,
        'start_time': final_range.start_time,
        'end_time': final_range.end_time,
        'source_classifications': list(final_range.source_classifications),
        'includes_retained_undetermined': final_range.includes_retained_undetermined,
        'retained_undetermined_count': final_range.retained_undetermined_count,
    }



def detect_staged_activity_ranges(
    video_path: Path,
    chapter_range: ChapterRange,
    settings: DetectorSettings,
    progress_callback: Callable[[int], None] | None = None,
    use_stage3_art_state_prototype: bool = False,
) -> tuple[list[tuple[int, int]], dict[str, list[dict[str, object]]]]:
    records = detect_movement_evidence_records(
        video_path=video_path,
        chapter_range=chapter_range,
        settings=settings,
        progress_callback=progress_callback,
    )
    movement_spans = build_stage1_movement_spans(records, settings)
    candidate_unions = build_stage2_candidate_unions(movement_spans)
    stage3_art_state_samples = (
        collect_stage3_art_state_samples(
            video_path=video_path,
            chapter_range=chapter_range,
            settings=settings,
        )
        if use_stage3_art_state_prototype
        else None
    )
    screened_candidate_unions = screen_stage3_candidate_unions(
        candidate_unions,
        records,
        sampled_frames=stage3_art_state_samples,
        chapter_range=chapter_range,
        settings=settings,
        use_art_state_prototype=use_stage3_art_state_prototype,
    )
    classified_time_slices = classify_stage4_time_slices(screened_candidate_unions, records)
    refined_sub_slices = refine_stage5_sub_slices(screened_candidate_unions, classified_time_slices, records)
    final_candidate_ranges = assemble_stage6_candidate_ranges(refined_sub_slices)

    debug_payload = {
        'movement_evidence_records': [serialize_movement_evidence_record(record) for record in records],
        'movement_spans': [serialize_movement_span(span) for span in movement_spans],
        'candidate_unions': [serialize_candidate_union(candidate_union) for candidate_union in candidate_unions],
        'screened_candidate_unions': [serialize_screened_candidate_union(screened_union) for screened_union in screened_candidate_unions],
        'classified_time_slices': [serialize_classified_time_slice(time_slice) for time_slice in classified_time_slices],
        'refined_sub_slices': [serialize_classified_time_slice(time_slice) for time_slice in refined_sub_slices],
        'final_candidate_ranges': [serialize_final_candidate_range(final_range) for final_range in final_candidate_ranges],
    }
    final_ranges = [(final_range.start_frame, final_range.end_frame) for final_range in final_candidate_ranges]
    return final_ranges, debug_payload



def write_staged_debug_artifacts(debug_stem: Path, debug_payload: dict[str, list[dict[str, object]]]) -> dict[str, Path]:
    debug_stem.parent.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}
    for section_name, items in debug_payload.items():
        output_path = debug_stem.with_name(f"{debug_stem.name}_{section_name}.json")
        output_path.write_text(json.dumps(items, indent=2), encoding='utf-8')
        output_paths[section_name] = output_path
    return output_paths


















































































