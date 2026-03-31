from __future__ import annotations

# ============================================================
# SECTION A - Imports And Reused Detection Primitives
# ============================================================

from collections import deque
from dataclasses import dataclass, replace
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
    build_persistent_change_mask,
    build_trail_mask,
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
STRONG_CONTINUITY_GAP_FRAMES = 10
MAX_UNION_GAP_FRAMES = FRAME_RATE
MAX_SPATIAL_DISTANCE_CELLS = 2
PER_POINT_MOVEMENT_EVIDENCE_THRESHOLD = 0.50
MOVEMENT_STRENGTH_CHANGE_WEIGHT = 0.40
MOVEMENT_STRENGTH_SPATIAL_WEIGHT = 0.30
MOVEMENT_STRENGTH_TEMPORAL_WEIGHT = 0.30
STAGE3_REFERENCE_WINDOW_FRAMES = 10
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
STAGE5_MIN_SUBDIVISION_FRAMES = 15
STAGE6_MIN_ROCKY_CLUSTER_FRAMES = FRAME_RATE
STAGE6_MIN_ROCKY_SLICE_COUNT = 4
STAGE6_MIN_ROCKY_CLUSTER_AVERAGE_EVIDENCE = 0.68


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


def screen_stage3_candidate_unions(
    candidate_unions: Iterable[CandidateUnion],
    records: Iterable[MovementEvidenceRecord],
) -> list[ScreenedCandidateUnion]:
    return [screen_candidate_union(candidate_union, records) for candidate_union in candidate_unions]

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

    if not slice_records or footprint_size == 0 or lasting_change_evidence_score <= STAGE4_INVALID_EVIDENCE_SCORE:
        classification = 'invalid'
        reason = 'weak_slice_activity'
    elif not reference_windows_reliable:
        classification = 'undetermined'
        reason = 'reference_windows_unreliable'
    elif (
        lasting_change_evidence_score >= STAGE4_VALID_EVIDENCE_SCORE
        and (
            contrast_score >= STAGE4_MIN_CONTRAST_SCORE
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
        if lasting_change_evidence_score >= STRONG_ACTIVE_REFERENCE_UNDETERMINED_FLOOR:
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
) -> ClassifiedTimeSlice:
    if (
        time_slice.classification == 'undetermined'
        and time_slice.footprint_size > 0
        and time_slice.lasting_change_evidence_score >= ROCKY_MINIMUM_SIZE_EVIDENCE_FLOOR
    ):
        return replace(time_slice, classification='rocky', reason='minimum_subdivision_size_reached')
    if (
        allow_active_reference_rescue
        and time_slice.classification == 'invalid'
        and time_slice.reason == 'reference_windows_too_active'
        and time_slice.footprint_size > 0
        and time_slice.lasting_change_evidence_score >= ACTIVE_REFERENCE_ROCKY_RESCUE_FLOOR
    ):
        return replace(time_slice, classification='rocky', reason='active_reference_minimum_size_reached')
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

    def refine_slice(time_slice: ClassifiedTimeSlice) -> list[ClassifiedTimeSlice]:
        allow_active_reference_rescue = time_slice.parent_union_index in unions_with_valid_slice
        slice_duration = time_slice.end_frame - time_slice.start_frame

        if time_slice.classification == 'invalid':
            if (
                allow_active_reference_rescue
                and time_slice.reason == 'reference_windows_too_active'
                and slice_duration <= minimum_subdivision_frames
            ):
                return [
                    classify_stage5_minimum_size_leaf(
                        time_slice,
                        allow_active_reference_rescue=True,
                    )
                ]
            if not (
                allow_active_reference_rescue
                and time_slice.reason == 'reference_windows_too_active'
                and slice_duration > minimum_subdivision_frames
            ):
                return [time_slice]

        elif time_slice.classification != 'undetermined':
            return [time_slice]

        if slice_duration <= minimum_subdivision_frames:
            return [
                classify_stage5_minimum_size_leaf(
                    time_slice,
                    allow_active_reference_rescue=allow_active_reference_rescue,
                )
            ]

        if slice_duration <= minimum_subdivision_frames:
            return [
                classify_stage5_minimum_size_leaf(
                    time_slice,
                    allow_active_reference_rescue=allow_active_reference_rescue,
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
                )
            ]

        refined_leaves: list[ClassifiedTimeSlice] = []
        for slice_index, (slice_start, slice_end) in enumerate(sub_slice_ranges, start=1):
            sub_slice_duration = slice_end - slice_start
            sub_slice_reference_window = max(
                minimum_subdivision_frames,
                min(STAGE3_REFERENCE_WINDOW_FRAMES, sub_slice_duration),
            )
            classified_sub_slice = classify_time_slice(
                screened_candidate_union=screened_candidate_union,
                slice_index=slice_index,
                slice_level=time_slice.slice_level + 1,
                slice_start=slice_start,
                slice_end=slice_end,
                records=ordered_records,
                reference_window_frames=sub_slice_reference_window,
            )
            classified_sub_slice = replace(
                classified_sub_slice,
                parent_range=(time_slice.start_frame, time_slice.end_frame),
            )
            refined_leaves.extend(refine_slice(classified_sub_slice))
        return refined_leaves

    refined_slices: list[ClassifiedTimeSlice] = []
    for time_slice in classified_time_slices:
        refined_slices.extend(refine_slice(time_slice))
    return refined_slices
# ============================================================
# SECTION I - Stage 6 Final Candidate Range Assembly
# ============================================================


def slices_are_adjacent(first_slice: ClassifiedTimeSlice, second_slice: ClassifiedTimeSlice) -> bool:
    return (
        first_slice.parent_union_index == second_slice.parent_union_index
        and first_slice.end_frame == second_slice.start_frame
    )



def is_stage6_candidate_slice(
    time_slice: ClassifiedTimeSlice,
    maximum_retained_frames: int = STAGE5_MIN_SUBDIVISION_FRAMES,
) -> bool:
    if time_slice.classification == 'valid':
        return True
    if time_slice.classification != 'rocky':
        return False
    if time_slice.reason not in {'minimum_subdivision_size_reached', 'active_reference_minimum_size_reached'}:
        return False
    return (time_slice.end_frame - time_slice.start_frame) <= maximum_retained_frames



def build_stage6_candidate_groups(
    refined_slices: Iterable[ClassifiedTimeSlice],
    maximum_retained_frames: int = STAGE5_MIN_SUBDIVISION_FRAMES,
    merge_gap_frames: int = 0,
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
    for next_slice in ordered_slices[1:]:
        previous_slice = current_group[-1]
        same_union = next_slice.parent_union_index == previous_slice.parent_union_index
        close_enough = (next_slice.start_frame - previous_slice.end_frame) <= merge_gap_frames
        if same_union and close_enough:
            current_group.append(next_slice)
        else:
            candidate_groups.append(current_group)
            current_group = [next_slice]
    candidate_groups.append(current_group)
    return candidate_groups



def should_keep_stage6_group(candidate_group: list[ClassifiedTimeSlice]) -> bool:
    if not candidate_group:
        return False
    if any(slice_info.classification == 'valid' for slice_info in candidate_group):
        return True

    total_duration = candidate_group[-1].end_frame - candidate_group[0].start_frame
    rocky_slice_count = len(candidate_group)
    average_rocky_evidence = sum(slice_info.lasting_change_evidence_score for slice_info in candidate_group) / rocky_slice_count
    contains_active_reference_rescue = any(
        slice_info.reason == 'active_reference_minimum_size_reached'
        for slice_info in candidate_group
    )
    standard_cluster_survival = (
        total_duration >= STAGE6_MIN_ROCKY_CLUSTER_FRAMES
        and rocky_slice_count >= STAGE6_MIN_ROCKY_SLICE_COUNT
        and average_rocky_evidence >= STAGE6_MIN_ROCKY_CLUSTER_AVERAGE_EVIDENCE
    )
    long_active_reference_survival = (
        contains_active_reference_rescue
        and total_duration >= (STAGE6_MIN_ROCKY_CLUSTER_FRAMES * 2)
        and rocky_slice_count >= STAGE6_MIN_ROCKY_SLICE_COUNT
        and average_rocky_evidence >= (ROCKY_MINIMUM_SIZE_EVIDENCE_FLOOR + 0.02)
    )
    return standard_cluster_survival or long_active_reference_survival


def assemble_stage6_candidate_ranges(
    refined_slices: Iterable[ClassifiedTimeSlice],
    retained_undetermined_max_frames: int = STAGE5_MIN_SUBDIVISION_FRAMES,
    merge_gap_frames: int = 0,
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
        retained_undetermined_count = sum(1 for slice_info in candidate_group if slice_info.classification == 'rocky')
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
) -> tuple[list[tuple[int, int]], dict[str, list[dict[str, object]]]]:
    records = detect_movement_evidence_records(
        video_path=video_path,
        chapter_range=chapter_range,
        settings=settings,
        progress_callback=progress_callback,
    )
    movement_spans = build_stage1_movement_spans(records, settings)
    candidate_unions = build_stage2_candidate_unions(movement_spans)
    screened_candidate_unions = screen_stage3_candidate_unions(candidate_unions, records)
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






















