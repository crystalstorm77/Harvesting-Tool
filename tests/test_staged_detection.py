from __future__ import annotations

# ============================================================
# SECTION A - Imports And Helpers
# ============================================================

import tempfile
import unittest
from dataclasses import replace
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from harvesting_tool.detection import ChapterRange, DetectorSettings, GRID_COLUMNS, GRID_ROWS, Timecode, extract_art_state_region
from harvesting_tool.staged_detection import (
    CandidateUnion,
    ClassifiedTimeSlice,
    MovementEvidenceRecord,
    MovementSpan,
    ScreenedCandidateUnion,
    TOTAL_GRID_BLOCKS,
    assemble_stage6_candidate_ranges,
    build_stage6_candidate_groups,
    build_movement_evidence_record,
    build_stage1_movement_spans,
    build_stage2_candidate_unions,
    build_staged_debug_summary_lines,
    decide_stage3_bucket_outcome,
    evaluate_stage4_cell_level_probe,
    find_stage4_near_global_opening_attribution_start_frame,
    stage4_extract_post_reset_contaminated_cells,
    load_precomputed_stage3_art_state_sample_cache_from_movement_evidence_path,
    load_reusable_stage3_art_state_sample_cache,
    build_stage4_bands_from_probe_results,
    classify_stage4_time_slices,
    classify_stage4_time_slices_with_subregion_debug,
    classify_stage5_minimum_size_leaf,
    find_stage5_broad_movement_boundary_frame,
    refine_stage5_sub_slices,
    screen_stage3_candidate_unions,
    select_stage3_art_state_reference_window,
    serialize_movement_evidence_record,
    split_footprint_into_local_subregions,
    write_reusable_stage3_art_state_sample_cache,
    write_staged_debug_artifacts,
)

def build_stage3_art_gray_for_cells(
    changed_cells: tuple[tuple[int, int], ...],
) -> np.ndarray:
    canvas_gray = np.zeros((120, 120), dtype=np.uint8)
    canvas_height, canvas_width = canvas_gray.shape[:2]
    cell_height = max(1, canvas_height // GRID_ROWS)
    cell_width = max(1, canvas_width // GRID_COLUMNS)

    for row_index, column_index in changed_cells:
        top = row_index * cell_height
        left = column_index * cell_width
        bottom = canvas_height if row_index == (GRID_ROWS - 1) else min(canvas_height, (row_index + 1) * cell_height)
        right = canvas_width if column_index == (GRID_COLUMNS - 1) else min(canvas_width, (column_index + 1) * cell_width)
        canvas_gray[top:bottom, left:right] = 255

    return extract_art_state_region(canvas_gray)

def make_stage3_art_state_sample(
    frame_index: int,
    *,
    changed: bool = False,
    changed_cells: tuple[tuple[int, int], ...] = (),
) -> dict[str, object]:
    canvas_gray = np.zeros((120, 120), dtype=np.uint8)
    if changed_cells:
        art_gray = build_stage3_art_gray_for_cells(changed_cells)
    else:
        if changed:
            canvas_gray[40:70, 40:70] = 255
        art_gray = extract_art_state_region(canvas_gray)
    return {
        'frame_index': frame_index,
        'canvas_shape': tuple(int(dimension) for dimension in canvas_gray.shape),
        'art_gray': art_gray,
    }

def make_settings() -> DetectorSettings:
    return DetectorSettings(
        lead_in=Timecode.from_seconds_and_frames(0, 2),
        tail_after=Timecode.from_seconds_and_frames(0, 4),
        min_harvest=Timecode.from_hhmmssff("00:00:05:00"),
        max_harvest=Timecode.from_hhmmssff("00:02:00:00"),
        min_clip_length=Timecode.from_hhmmssff("00:00:00:15"),
        max_clip_length=Timecode.from_hhmmssff("00:00:07:00"),
        pause_threshold=Timecode.from_hhmmssff("00:00:05:00"),
        min_burst_length=Timecode.from_hhmmssff("00:00:00:06"),
        sample_stride=2,
        activity_threshold=8.0,
        active_pixel_ratio=0.0015,
    )


def make_record(
    record_index: int,
    frame_index: int,
    *,
    movement_present: bool = True,
    opening_signal: bool = False,
    continuation_signal: bool = False,
    weak_signal: bool = False,
    touched_grid_coordinates: tuple[tuple[int, int], ...] = (),
    movement_strength_score: float | None = None,
    temporal_persistence_score: float | None = None,
    spatial_extent_score: float | None = None,
) -> MovementEvidenceRecord:
    touched_count = len(touched_grid_coordinates)
    return MovementEvidenceRecord(
        record_index=record_index,
        evaluation_point_timecode=Timecode(total_frames=frame_index).to_hhmmssff(),
        frame_index=frame_index,
        movement_present=movement_present,
        touched_grid_coordinates=touched_grid_coordinates,
        touched_grid_coordinate_count=touched_count,
        change_magnitude_score=movement_strength_score if movement_strength_score is not None else (0.5 if movement_present else 0.0),
        spatial_extent_score=spatial_extent_score if spatial_extent_score is not None else min(1.0, touched_count / 4.0),
        temporal_persistence_score=temporal_persistence_score if temporal_persistence_score is not None else (0.5 if movement_present else 0.0),
        movement_strength_score=movement_strength_score if movement_strength_score is not None else (0.5 if movement_present else 0.0),
        opening_signal=opening_signal,
        continuation_signal=continuation_signal,
        weak_signal=weak_signal,
    )


def make_screened_union(
    *,
    union_index: int = 1,
    start_frame: int = 300,
    end_frame: int = 360,
    union_footprint_size: int = 4,
    union_footprint: frozenset[tuple[int, int]] | None = None,
    surviving: bool = True,
    provisional_survival: bool = False,
    screening_result: str = "surviving",
) -> ScreenedCandidateUnion:
    resolved_union_footprint = (
        union_footprint
        if union_footprint is not None
        else (frozenset({(4, 4), (4, 5), (5, 4), (5, 5)}) if union_footprint_size else frozenset())
    )
    candidate_union = CandidateUnion(
        union_index=union_index,
        start_frame=start_frame,
        end_frame=end_frame,
        start_time=Timecode(total_frames=start_frame).to_hhmmssff(),
        end_time=Timecode(total_frames=end_frame).to_hhmmssff(),
        member_movement_spans=(),
        union_footprint=resolved_union_footprint,
        union_footprint_size=len(resolved_union_footprint),
    )
    return ScreenedCandidateUnion(
        candidate_union=candidate_union,
        screening_result=screening_result,
        surviving=surviving,
        provisional_survival=provisional_survival,
        reason="fixture",
        within_union_record_count=0,
        before_record_count=0,
        after_record_count=0,
        mean_movement_strength=0.0,
        mean_temporal_persistence=0.0,
        mean_spatial_extent=0.0,
        lasting_change_evidence_score=0.0,
        before_reference_activity=0.0,
        after_reference_activity=0.0,
        reference_windows_reliable=True,
    )

# ============================================================
# SECTION B - Record Scoring Tests
# ============================================================


class MovementEvidenceRecordTests(unittest.TestCase):
    def test_build_movement_evidence_record_scores_and_grid_coordinates(self) -> None:
        settings = make_settings()
        previous = np.zeros((120, 120), dtype=np.uint8)
        current = previous.copy()
        next_frame = previous.copy()
        current[20:40, 30:50] = 255
        next_frame[20:40, 30:50] = 255

        record = build_movement_evidence_record(
            record_index=1,
            previous_sample={"frame_index": 10, "gray": previous},
            current_sample={"frame_index": 12, "gray": current},
            next_sample={"frame_index": 14, "gray": next_frame},
            recent_persistent_masks=deque(maxlen=4),
            settings=settings,
            cv2=cv2,
        )

        self.assertTrue(record.movement_present)
        self.assertGreater(record.touched_grid_coordinate_count, 0)
        self.assertGreater(record.change_magnitude_score, 0.0)
        self.assertGreater(record.temporal_persistence_score, 0.0)
        self.assertGreater(record.movement_strength_score, 0.0)


# ============================================================
# SECTION C - Stage 1 Movement Span Tests
# ============================================================


class MovementSpanConstructionTests(unittest.TestCase):
    def test_stage1_movement_spans_open_when_two_of_last_three_records_are_active(self) -> None:
        settings = make_settings()
        records = [
            make_record(1, 100, movement_present=True, touched_grid_coordinates=((0, 0),)),
            make_record(2, 102, movement_present=False),
            make_record(3, 104, movement_present=True, touched_grid_coordinates=((1, 1),)),
            make_record(4, 106, movement_present=True, touched_grid_coordinates=((1, 2),)),
            make_record(5, 108, movement_present=False),
            make_record(6, 110, movement_present=False),
            make_record(7, 112, movement_present=False),
        ]

        spans = build_stage1_movement_spans(records, settings)

        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertEqual(span.start_frame, 100)
        self.assertEqual(span.end_frame, 108)
        self.assertEqual(span.record_indices, (1, 3, 4))
        self.assertEqual(
            span.footprint,
            frozenset({(0, 0), (1, 1), (1, 2)}),
        )

    def test_stage1_movement_spans_allow_brief_inactive_dips_without_closing(self) -> None:
        settings = make_settings()
        records = [
            make_record(1, 200, movement_present=True, touched_grid_coordinates=((2, 2),)),
            make_record(2, 202, movement_present=False),
            make_record(3, 204, movement_present=True, touched_grid_coordinates=((2, 3),)),
            make_record(4, 206, movement_present=False),
            make_record(5, 208, movement_present=True, touched_grid_coordinates=((3, 3),)),
            make_record(6, 210, movement_present=False),
            make_record(7, 212, movement_present=False),
            make_record(8, 214, movement_present=False),
        ]

        spans = build_stage1_movement_spans(records, settings)

        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertEqual(span.start_frame, 200)
        self.assertEqual(span.end_frame, 210)
        self.assertEqual(span.record_indices, (1, 3, 5))

    def test_stage1_movement_spans_do_not_open_on_single_isolated_active_record(self) -> None:
        settings = make_settings()
        records = [
            make_record(1, 300, movement_present=False),
            make_record(2, 302, movement_present=True, touched_grid_coordinates=((4, 4),)),
            make_record(3, 304, movement_present=False),
            make_record(4, 306, movement_present=False),
        ]

        spans = build_stage1_movement_spans(records, settings)
        self.assertEqual(spans, [])

    def test_stage1_movement_spans_discard_short_runs_below_minimum_length(self) -> None:
        settings = make_settings()
        records = [
            make_record(1, 400, movement_present=True, touched_grid_coordinates=((3, 3),)),
            make_record(2, 402, movement_present=True, touched_grid_coordinates=((3, 4),)),
            make_record(3, 404, movement_present=False),
            make_record(4, 406, movement_present=False),
            make_record(5, 408, movement_present=False),
        ]

        spans = build_stage1_movement_spans(records, settings)
        self.assertEqual(spans, [])
# ============================================================
# SECTION D - Stage 2 Candidate Union Tests
# ============================================================


class CandidateUnionConstructionTests(unittest.TestCase):
    def test_stage2_candidate_unions_merge_on_short_temporal_gap_with_spatial_support(self) -> None:
        spans = [
            MovementSpan(
                span_index=1,
                start_frame=300,
                end_frame=330,
                start_time="00:00:10:00",
                end_time="00:00:11:00",
                footprint=frozenset({(2, 2)}),
                footprint_size=1,
                record_indices=(1, 2),
            ),
            MovementSpan(
                span_index=2,
                start_frame=336,
                end_frame=360,
                start_time="00:00:11:06",
                end_time="00:00:12:00",
                footprint=frozenset({(3, 2)}),
                footprint_size=1,
                record_indices=(3, 4),
            ),
        ]

        unions = build_stage2_candidate_unions(spans)

        self.assertEqual(len(unions), 1)
        self.assertEqual(unions[0].member_movement_spans, tuple(spans))
        self.assertEqual(unions[0].union_footprint_size, 2)

    def test_stage2_candidate_unions_auto_merge_on_strong_continuity_gap_without_spatial_support(self) -> None:
        spans = [
            MovementSpan(
                span_index=1,
                start_frame=300,
                end_frame=330,
                start_time="00:00:10:00",
                end_time="00:00:11:00",
                footprint=frozenset({(1, 1)}),
                footprint_size=1,
                record_indices=(1, 2),
            ),
            MovementSpan(
                span_index=2,
                start_frame=338,
                end_frame=360,
                start_time="00:00:11:08",
                end_time="00:00:12:00",
                footprint=frozenset({(10, 10)}),
                footprint_size=1,
                record_indices=(3, 4),
            ),
        ]

        unions = build_stage2_candidate_unions(spans)

        self.assertEqual(len(unions), 1)
        self.assertEqual(unions[0].member_movement_spans, tuple(spans))

    def test_stage2_candidate_unions_split_without_spatial_support_outside_strong_continuity_gap(self) -> None:
        spans = [
            MovementSpan(
                span_index=1,
                start_frame=300,
                end_frame=330,
                start_time="00:00:10:00",
                end_time="00:00:11:00",
                footprint=frozenset({(2, 2)}),
                footprint_size=1,
                record_indices=(1, 2),
            ),
            MovementSpan(
                span_index=2,
                start_frame=348,
                end_frame=360,
                start_time="00:00:11:18",
                end_time="00:00:12:00",
                footprint=frozenset({(8, 8)}),
                footprint_size=1,
                record_indices=(3, 4),
            ),
        ]

        unions = build_stage2_candidate_unions(spans)

        self.assertEqual(len(unions), 2)
        self.assertEqual(unions[0].member_movement_spans, (spans[0],))
        self.assertEqual(unions[1].member_movement_spans, (spans[1],))

    def test_stage2_candidate_unions_merge_on_spatial_support_within_union_gap(self) -> None:
        spans = [
            MovementSpan(
                span_index=1,
                start_frame=300,
                end_frame=330,
                start_time="00:00:10:00",
                end_time="00:00:11:00",
                footprint=frozenset({(4, 4)}),
                footprint_size=1,
                record_indices=(1, 2),
            ),
            MovementSpan(
                span_index=2,
                start_frame=348,
                end_frame=378,
                start_time="00:00:11:18",
                end_time="00:00:12:18",
                footprint=frozenset({(5, 5)}),
                footprint_size=1,
                record_indices=(3, 4),
            ),
        ]

        unions = build_stage2_candidate_unions(spans)

        self.assertEqual(len(unions), 1)
        self.assertEqual(unions[0].union_footprint, frozenset({(4, 4), (5, 5)}))

    def test_stage2_candidate_unions_split_without_spatial_support(self) -> None:
        spans = [
            MovementSpan(
                span_index=1,
                start_frame=300,
                end_frame=330,
                start_time="00:00:10:00",
                end_time="00:00:11:00",
                footprint=frozenset({(0, 0)}),
                footprint_size=1,
                record_indices=(1, 2),
            ),
            MovementSpan(
                span_index=2,
                start_frame=348,
                end_frame=378,
                start_time="00:00:11:18",
                end_time="00:00:12:18",
                footprint=frozenset({(9, 9)}),
                footprint_size=1,
                record_indices=(3, 4),
            ),
            MovementSpan(
                span_index=3,
                start_frame=402,
                end_frame=432,
                start_time="00:00:13:12",
                end_time="00:00:14:12",
                footprint=frozenset({(9, 10)}),
                footprint_size=1,
                record_indices=(5, 6),
            ),
        ]

        unions = build_stage2_candidate_unions(spans)

        self.assertEqual(len(unions), 2)
        self.assertEqual(unions[0].member_movement_spans, (spans[0],))
        self.assertEqual(unions[1].member_movement_spans, (spans[1], spans[2]))

    def test_stage2_candidate_unions_allow_normal_local_growth(self) -> None:
        spans = [
            MovementSpan(
                span_index=1,
                start_frame=300,
                end_frame=330,
                start_time="00:00:10:00",
                end_time="00:00:11:00",
                footprint=frozenset({(4, 4), (4, 5), (5, 4), (5, 5), (6, 4), (6, 5)}),
                footprint_size=6,
                record_indices=(1, 2),
            ),
            MovementSpan(
                span_index=2,
                start_frame=342,
                end_frame=372,
                start_time="00:00:11:12",
                end_time="00:00:12:12",
                footprint=frozenset({(4, 5), (5, 5), (5, 6), (6, 5), (6, 6), (7, 5), (7, 6)}),
                footprint_size=7,
                record_indices=(3, 4),
            ),
        ]

        unions = build_stage2_candidate_unions(spans)

        self.assertEqual(len(unions), 1)
        self.assertEqual(unions[0].member_movement_spans, tuple(spans))

    def test_stage2_candidate_unions_allow_large_growth_when_new_cells_remain_well_attached(self) -> None:
        spans = [
            MovementSpan(
                span_index=1,
                start_frame=300,
                end_frame=330,
                start_time="00:00:10:00",
                end_time="00:00:11:00",
                footprint=frozenset({(4, 4), (4, 5), (5, 4), (5, 5)}),
                footprint_size=4,
                record_indices=(1, 2),
            ),
            MovementSpan(
                span_index=2,
                start_frame=342,
                end_frame=372,
                start_time="00:00:11:12",
                end_time="00:00:12:12",
                footprint=frozenset({(5, 5), (5, 6), (6, 5), (6, 6)}),
                footprint_size=4,
                record_indices=(3, 4),
            ),
            MovementSpan(
                span_index=3,
                start_frame=384,
                end_frame=414,
                start_time="00:00:12:24",
                end_time="00:00:13:24",
                footprint=frozenset({(4, 4), (4, 5), (4, 6), (5, 4), (5, 5), (5, 6), (6, 4), (6, 5), (6, 6), (7, 4), (7, 5), (7, 6), (8, 4), (8, 5), (8, 6)}),
                footprint_size=15,
                record_indices=(5, 6),
            ),
        ]

        unions = build_stage2_candidate_unions(spans)

        self.assertEqual(len(unions), 1)
        self.assertEqual(unions[0].member_movement_spans, tuple(spans))

    def test_stage2_candidate_unions_split_when_weak_path_growth_is_large_and_poorly_attached(self) -> None:
        spans = [
            MovementSpan(
                span_index=1,
                start_frame=300,
                end_frame=330,
                start_time="00:00:10:00",
                end_time="00:00:11:00",
                footprint=frozenset({(4, 4), (4, 5), (5, 4), (5, 5)}),
                footprint_size=4,
                record_indices=(1, 2),
            ),
            MovementSpan(
                span_index=2,
                start_frame=348,
                end_frame=378,
                start_time="00:00:11:18",
                end_time="00:00:12:18",
                footprint=frozenset({(4, 4), (4, 5), (5, 4), (5, 5)}),
                footprint_size=4,
                record_indices=(3, 4),
            ),
            MovementSpan(
                span_index=3,
                start_frame=402,
                end_frame=432,
                start_time="00:00:13:12",
                end_time="00:00:14:12",
                footprint=frozenset({(5, 5), (9, 9), (9, 10), (10, 9), (10, 10), (11, 9), (11, 10)}),
                footprint_size=7,
                record_indices=(5, 6),
            ),
        ]

        unions = build_stage2_candidate_unions(spans)

        self.assertEqual(len(unions), 2)
        self.assertEqual(unions[0].member_movement_spans, (spans[0], spans[1]))
        self.assertEqual(unions[1].member_movement_spans, (spans[2],))
# ============================================================
# SECTION E - Stage 3 Candidate Union Screening Tests
# ============================================================


class CandidateUnionScreeningTests(unittest.TestCase):
    def test_stage3_bucket_outcome_survives_diluted_union_via_local_changed_cluster(self) -> None:
        screening_result, surviving, reason, bounded_ambiguity = decide_stage3_bucket_outcome(
            coverage_summary={
                'total_footprint_cells': 64,
                'resolved_changed_coverage': 0.078125,
                'resolved_unchanged_coverage': 0.671875,
                'resolved_ambiguous_coverage': 0.25,
                'largest_changed_cluster_size': 5,
                'largest_changed_cluster_coverage': 0.078125,
                'ambiguous_cluster_count': 2,
            },
            reconstructed_before_coverage=0.92,
            mode='snapshot_rescue',
        )

        self.assertEqual(screening_result, 'surviving')
        self.assertTrue(surviving)
        self.assertEqual(reason, 'survived_by_local_changed_cluster')
        self.assertFalse(bounded_ambiguity)

    def test_stage3_bucket_outcome_does_not_rescue_tiny_changed_cluster(self) -> None:
        screening_result, surviving, reason, bounded_ambiguity = decide_stage3_bucket_outcome(
            coverage_summary={
                'total_footprint_cells': 64,
                'resolved_changed_coverage': 0.03125,
                'resolved_unchanged_coverage': 0.76875,
                'resolved_ambiguous_coverage': 0.20,
                'largest_changed_cluster_size': 2,
                'largest_changed_cluster_coverage': 0.03125,
                'ambiguous_cluster_count': 2,
            },
            reconstructed_before_coverage=0.92,
            mode='snapshot_rescue',
        )

        self.assertEqual(screening_result, 'rejected')
        self.assertFalse(surviving)
        self.assertEqual(reason, 'rejected_after_rescue_failure')
        self.assertFalse(bounded_ambiguity)

    def test_stage3_candidate_union_rejects_weak_union_activity(self) -> None:
        candidate_union = CandidateUnion(
            union_index=1,
            start_frame=300,
            end_frame=330,
            start_time="00:00:10:00",
            end_time="00:00:11:00",
            member_movement_spans=(),
            union_footprint=frozenset({(2, 2)}),
            union_footprint_size=1,
        )
        records = [
            make_record(1, 304, movement_present=True, movement_strength_score=0.08, temporal_persistence_score=0.08, spatial_extent_score=0.05),
            make_record(2, 316, movement_present=True, movement_strength_score=0.09, temporal_persistence_score=0.07, spatial_extent_score=0.05),
            make_record(3, 336, movement_present=False, movement_strength_score=0.01, temporal_persistence_score=0.01),
            make_record(4, 348, movement_present=False, movement_strength_score=0.01, temporal_persistence_score=0.01),
        ]

        screened = screen_stage3_candidate_unions([candidate_union], records)

        self.assertEqual(screened[0].screening_result, "rejected")
        self.assertEqual(screened[0].reason, "weak_union_activity")

    def test_stage3_step1_clear_survival_uses_full_footprint_comparison(self) -> None:
        candidate_union = CandidateUnion(
            union_index=1,
            start_frame=300,
            end_frame=330,
            start_time='00:00:10:00',
            end_time='00:00:11:00',
            member_movement_spans=(),
            union_footprint=frozenset({(4, 4), (4, 5), (5, 4), (5, 5)}),
            union_footprint_size=4,
        )
        records = [
            make_record(1, 304, movement_present=True, movement_strength_score=0.62, temporal_persistence_score=0.58, spatial_extent_score=0.5, touched_grid_coordinates=((4, 5), (4, 6), (5, 5), (5, 6))),
            make_record(2, 316, movement_present=True, movement_strength_score=0.66, temporal_persistence_score=0.60, spatial_extent_score=0.5, touched_grid_coordinates=((4, 5), (4, 6), (5, 5), (5, 6))),
            make_record(3, 338, movement_present=True, movement_strength_score=0.30, temporal_persistence_score=0.28, spatial_extent_score=0.10, touched_grid_coordinates=((0, 0),)),
        ]
        chapter_range = ChapterRange(
            start=Timecode.from_hhmmssff('00:00:09:00'),
            end=Timecode.from_hhmmssff('00:00:20:00'),
        )
        sampled_frames = [
            make_stage3_art_state_sample(285, changed=False),
            make_stage3_art_state_sample(290, changed=False),
            make_stage3_art_state_sample(336, changed=True),
            make_stage3_art_state_sample(340, changed=True),
        ]

        screened = screen_stage3_candidate_unions(
            [candidate_union],
            records,
            sampled_frames=sampled_frames,
            chapter_range=chapter_range,
            settings=make_settings(),
        )

        self.assertEqual(screened[0].screening_result, 'surviving')
        self.assertEqual(screened[0].reason, 'step1_clear_survival')
        self.assertEqual(screened[0].stage3_mode, 'step1')
        self.assertEqual(screened[0].stage3_alignment_mode, 'full_footprint')
        self.assertGreaterEqual(screened[0].lasting_change_evidence_score, 0.10)

    def test_stage3_step1_clear_rejection_rejects_unchanged_union(self) -> None:
        candidate_union = CandidateUnion(
            union_index=1,
            start_frame=300,
            end_frame=330,
            start_time='00:00:10:00',
            end_time='00:00:11:00',
            member_movement_spans=(),
            union_footprint=frozenset({(4, 4), (4, 5), (5, 4), (5, 5)}),
            union_footprint_size=4,
        )
        records = [
            make_record(1, 304, movement_present=True, movement_strength_score=0.62, temporal_persistence_score=0.58, spatial_extent_score=0.5, touched_grid_coordinates=((4, 5), (4, 6), (5, 5), (5, 6))),
            make_record(2, 316, movement_present=True, movement_strength_score=0.66, temporal_persistence_score=0.60, spatial_extent_score=0.5, touched_grid_coordinates=((4, 5), (4, 6), (5, 5), (5, 6))),
            make_record(3, 338, movement_present=True, movement_strength_score=0.30, temporal_persistence_score=0.28, spatial_extent_score=0.10, touched_grid_coordinates=((0, 0),)),
        ]
        chapter_range = ChapterRange(
            start=Timecode.from_hhmmssff('00:00:09:00'),
            end=Timecode.from_hhmmssff('00:00:20:00'),
        )
        sampled_frames = [
            make_stage3_art_state_sample(285, changed=False),
            make_stage3_art_state_sample(290, changed=False),
            make_stage3_art_state_sample(336, changed=False),
            make_stage3_art_state_sample(340, changed=False),
        ]

        screened = screen_stage3_candidate_unions(
            [candidate_union],
            records,
            sampled_frames=sampled_frames,
            chapter_range=chapter_range,
            settings=make_settings(),
        )

        self.assertEqual(screened[0].screening_result, 'rejected')
        self.assertEqual(screened[0].reason, 'step1_clear_rejection')
        self.assertEqual(screened[0].stage3_mode, 'step1')

    def test_stage3_step1_endpoint_cells_do_not_accept_idle_after_window_without_external_movement(self) -> None:
        candidate_union = CandidateUnion(
            union_index=1,
            start_frame=300,
            end_frame=330,
            start_time='00:00:10:00',
            end_time='00:00:11:00',
            member_movement_spans=(),
            union_footprint=frozenset({(4, 4), (4, 5), (5, 4), (5, 5)}),
            union_footprint_size=4,
        )
        records = [
            make_record(1, 304, movement_present=True, movement_strength_score=0.62, temporal_persistence_score=0.58, spatial_extent_score=0.5, touched_grid_coordinates=((4, 5), (4, 6), (5, 5), (5, 6))),
            make_record(2, 316, movement_present=True, movement_strength_score=0.66, temporal_persistence_score=0.60, spatial_extent_score=0.5, touched_grid_coordinates=((4, 5), (4, 6), (5, 5), (5, 6))),
        ]
        chapter_range = ChapterRange(
            start=Timecode.from_hhmmssff('00:00:09:00'),
            end=Timecode.from_hhmmssff('00:00:20:00'),
        )
        sampled_frames = [
            make_stage3_art_state_sample(285, changed=False),
            make_stage3_art_state_sample(290, changed=False),
            make_stage3_art_state_sample(336, changed=True),
            make_stage3_art_state_sample(340, changed=True),
        ]

        screened = screen_stage3_candidate_unions(
            [candidate_union],
            records,
            sampled_frames=sampled_frames,
            chapter_range=chapter_range,
            settings=make_settings(),
        )

        self.assertEqual(screened[0].screening_result, 'rejected')
        self.assertEqual(screened[0].reason, 'rejected_after_rescue_failure')
        self.assertEqual(screened[0].stage3_mode, 'snapshot_rescue')

    def test_stage3_step1_endpoint_cells_accept_after_window_when_movement_continues_elsewhere(self) -> None:
        candidate_union = CandidateUnion(
            union_index=1,
            start_frame=300,
            end_frame=330,
            start_time='00:00:10:00',
            end_time='00:00:11:00',
            member_movement_spans=(),
            union_footprint=frozenset({(4, 4), (4, 5), (5, 4), (5, 5)}),
            union_footprint_size=4,
        )
        records = [
            make_record(1, 304, movement_present=True, movement_strength_score=0.62, temporal_persistence_score=0.58, spatial_extent_score=0.5, touched_grid_coordinates=((4, 5), (4, 6), (5, 5), (5, 6))),
            make_record(2, 316, movement_present=True, movement_strength_score=0.66, temporal_persistence_score=0.60, spatial_extent_score=0.5, touched_grid_coordinates=((4, 5), (4, 6), (5, 5), (5, 6))),
            make_record(3, 338, movement_present=True, movement_strength_score=0.30, temporal_persistence_score=0.28, spatial_extent_score=0.10, touched_grid_coordinates=((0, 0),)),
        ]
        chapter_range = ChapterRange(
            start=Timecode.from_hhmmssff('00:00:09:00'),
            end=Timecode.from_hhmmssff('00:00:20:00'),
        )
        sampled_frames = [
            make_stage3_art_state_sample(285, changed=False),
            make_stage3_art_state_sample(290, changed=False),
            make_stage3_art_state_sample(336, changed=True),
            make_stage3_art_state_sample(340, changed=True),
        ]

        screened = screen_stage3_candidate_unions(
            [candidate_union],
            records,
            sampled_frames=sampled_frames,
            chapter_range=chapter_range,
            settings=make_settings(),
        )

        self.assertEqual(screened[0].screening_result, 'surviving')
        self.assertEqual(screened[0].reason, 'step1_clear_survival')
        self.assertEqual(screened[0].stage3_mode, 'step1')

    def test_stage3_snapshot_rescue_stitches_composite_before_state(self) -> None:
        candidate_union = CandidateUnion(
            union_index=1,
            start_frame=300,
            end_frame=330,
            start_time='00:00:10:00',
            end_time='00:00:11:00',
            member_movement_spans=(),
            union_footprint=frozenset({(4, 4), (4, 5), (5, 4), (5, 5)}),
            union_footprint_size=4,
        )
        records = [
            make_record(1, 278, movement_present=True, movement_strength_score=0.30, temporal_persistence_score=0.28, spatial_extent_score=0.20, touched_grid_coordinates=((4, 4), (4, 5))),
            make_record(2, 288, movement_present=True, movement_strength_score=0.30, temporal_persistence_score=0.28, spatial_extent_score=0.20, touched_grid_coordinates=((5, 4), (5, 5))),
            make_record(3, 304, movement_present=True, movement_strength_score=0.62, temporal_persistence_score=0.58, spatial_extent_score=0.5, touched_grid_coordinates=((4, 5), (4, 6), (5, 5), (5, 6))),
            make_record(4, 316, movement_present=True, movement_strength_score=0.66, temporal_persistence_score=0.60, spatial_extent_score=0.5, touched_grid_coordinates=((4, 5), (4, 6), (5, 5), (5, 6))),
            make_record(5, 338, movement_present=True, movement_strength_score=0.30, temporal_persistence_score=0.28, spatial_extent_score=0.10, touched_grid_coordinates=((0, 0),)),
        ]
        chapter_range = ChapterRange(
            start=Timecode.from_hhmmssff('00:00:09:02'),
            end=Timecode.from_hhmmssff('00:00:20:00'),
        )
        sampled_frames = [
            make_stage3_art_state_sample(272, changed=False),
            make_stage3_art_state_sample(276, changed=False),
            make_stage3_art_state_sample(280, changed=False),
            make_stage3_art_state_sample(284, changed=False),
            make_stage3_art_state_sample(288, changed=False),
            make_stage3_art_state_sample(292, changed=False),
            make_stage3_art_state_sample(336, changed=True),
            make_stage3_art_state_sample(340, changed=True),
        ]

        screened = screen_stage3_candidate_unions(
            [candidate_union],
            records,
            sampled_frames=sampled_frames,
            chapter_range=chapter_range,
            settings=make_settings(),
        )

        self.assertEqual(screened[0].screening_result, 'surviving')
        self.assertEqual(screened[0].reason, 'step1_clear_survival')
        self.assertEqual(screened[0].stage3_mode, 'step1')
        self.assertEqual(screened[0].stage3_alignment_mode, 'partial_footprint')
        self.assertGreaterEqual(screened[0].stage3_footprint_support_score, 0.80)

    def test_stage3_snapshot_rescue_can_survive_with_bounded_ambiguity(self) -> None:
        candidate_union = CandidateUnion(
            union_index=1,
            start_frame=300,
            end_frame=330,
            start_time='00:00:10:00',
            end_time='00:00:11:00',
            member_movement_spans=(),
            union_footprint=frozenset({(4, 4), (4, 5), (4, 6), (4, 7), (5, 4), (5, 5), (5, 6), (5, 7), (6, 4), (6, 5)}),
            union_footprint_size=10,
        )
        records = [
            make_record(1, 286, movement_present=True, movement_strength_score=0.20, temporal_persistence_score=0.18, spatial_extent_score=0.10, touched_grid_coordinates=((0, 0),)),
            make_record(2, 304, movement_present=True, movement_strength_score=0.62, temporal_persistence_score=0.58, spatial_extent_score=0.5, touched_grid_coordinates=((4, 4), (4, 5), (4, 6), (4, 7), (5, 4), (5, 5), (5, 6), (5, 7), (6, 4), (6, 5))),
            make_record(3, 316, movement_present=True, movement_strength_score=0.66, temporal_persistence_score=0.60, spatial_extent_score=0.5, touched_grid_coordinates=((4, 4), (4, 5), (4, 6), (4, 7), (5, 4), (5, 5), (5, 6), (5, 7), (6, 4), (6, 5))),
            make_record(4, 342, movement_present=True, movement_strength_score=0.30, temporal_persistence_score=0.28, spatial_extent_score=0.10, touched_grid_coordinates=((6, 5),)),
            make_record(5, 346, movement_present=True, movement_strength_score=0.30, temporal_persistence_score=0.28, spatial_extent_score=0.10, touched_grid_coordinates=((0, 0),)),
        ]
        chapter_range = ChapterRange(
            start=Timecode.from_hhmmssff('00:00:09:00'),
            end=Timecode.from_hhmmssff('00:00:11:20'),
        )
        sampled_frames = [
            make_stage3_art_state_sample(285, changed=False),
            make_stage3_art_state_sample(290, changed=False),
            make_stage3_art_state_sample(336, changed=False),
            make_stage3_art_state_sample(340, changed=False),
            make_stage3_art_state_sample(344, changed=False),
            make_stage3_art_state_sample(348, changed=False),
        ]

        screened = screen_stage3_candidate_unions(
            [candidate_union],
            records,
            sampled_frames=sampled_frames,
            chapter_range=chapter_range,
            settings=make_settings(),
        )

        self.assertEqual(screened[0].screening_result, 'surviving')
        self.assertEqual(screened[0].reason, 'survived_by_bounded_ambiguity')
        self.assertEqual(screened[0].stage3_mode, 'step1')
        self.assertLessEqual(1.0 - screened[0].stage3_after_window_persistence_score, 0.10)

    def test_stage3_snapshot_rescue_can_use_union_internal_after_state_when_post_union_windows_are_blocked(self) -> None:
        first_union = CandidateUnion(
            union_index=1,
            start_frame=300,
            end_frame=330,
            start_time='00:00:10:00',
            end_time='00:00:11:00',
            member_movement_spans=(),
            union_footprint=frozenset({(4, 4), (4, 5), (5, 4), (5, 5)}),
            union_footprint_size=4,
        )
        second_union = CandidateUnion(
            union_index=2,
            start_frame=332,
            end_frame=360,
            start_time='00:00:11:02',
            end_time='00:00:12:00',
            member_movement_spans=(),
            union_footprint=frozenset({(0, 0)}),
            union_footprint_size=1,
        )
        records = [
            make_record(1, 285, movement_present=True, movement_strength_score=0.20, temporal_persistence_score=0.18, spatial_extent_score=0.10, touched_grid_coordinates=((0, 1),)),
            make_record(2, 304, movement_present=True, movement_strength_score=0.62, temporal_persistence_score=0.58, spatial_extent_score=0.5, touched_grid_coordinates=((4, 4), (4, 5))),
            make_record(3, 316, movement_present=True, movement_strength_score=0.66, temporal_persistence_score=0.60, spatial_extent_score=0.5, touched_grid_coordinates=((5, 4), (5, 5))),
            make_record(4, 336, movement_present=True, movement_strength_score=0.30, temporal_persistence_score=0.28, spatial_extent_score=0.10, touched_grid_coordinates=((0, 0),)),
        ]
        chapter_range = ChapterRange(
            start=Timecode.from_hhmmssff('00:00:09:00'),
            end=Timecode.from_hhmmssff('00:00:20:00'),
        )
        sampled_frames = [
            make_stage3_art_state_sample(285, changed=False),
            make_stage3_art_state_sample(290, changed=False),
            make_stage3_art_state_sample(308, changed=True),
            make_stage3_art_state_sample(312, changed=True),
            make_stage3_art_state_sample(336, changed=True),
            make_stage3_art_state_sample(340, changed=True),
        ]

        screened = screen_stage3_candidate_unions(
            [first_union, second_union],
            records,
            sampled_frames=sampled_frames,
            chapter_range=chapter_range,
            settings=make_settings(),
        )

        self.assertEqual(screened[0].screening_result, 'surviving')
        self.assertEqual(screened[0].reason, 'survived_by_changed_evidence')
        self.assertEqual(screened[0].stage3_mode, 'snapshot_rescue')
        self.assertEqual(screened[0].stage3_alignment_mode, 'composite_before_after')
    def test_stage3_anti_borrowing_blocks_later_union_from_validating_current_union(self) -> None:
        first_union = CandidateUnion(
            union_index=1,
            start_frame=300,
            end_frame=330,
            start_time='00:00:10:00',
            end_time='00:00:11:00',
            member_movement_spans=(),
            union_footprint=frozenset({(4, 4), (4, 5), (5, 4), (5, 5)}),
            union_footprint_size=4,
        )
        second_union = CandidateUnion(
            union_index=2,
            start_frame=360,
            end_frame=390,
            start_time='00:00:12:00',
            end_time='00:00:13:00',
            member_movement_spans=(),
            union_footprint=frozenset({(4, 4), (4, 5), (5, 4), (5, 5)}),
            union_footprint_size=4,
        )
        records = [
            make_record(1, 304, movement_present=True, movement_strength_score=0.62, temporal_persistence_score=0.58, spatial_extent_score=0.5, touched_grid_coordinates=((4, 5), (4, 6), (5, 5), (5, 6))),
            make_record(2, 316, movement_present=True, movement_strength_score=0.66, temporal_persistence_score=0.60, spatial_extent_score=0.5, touched_grid_coordinates=((4, 5), (4, 6), (5, 5), (5, 6))),
            make_record(3, 364, movement_present=True, movement_strength_score=0.62, temporal_persistence_score=0.58, spatial_extent_score=0.5, touched_grid_coordinates=((4, 5), (4, 6), (5, 5), (5, 6))),
            make_record(4, 376, movement_present=True, movement_strength_score=0.66, temporal_persistence_score=0.60, spatial_extent_score=0.5, touched_grid_coordinates=((4, 5), (4, 6), (5, 5), (5, 6))),
        ]
        chapter_range = ChapterRange(
            start=Timecode.from_hhmmssff('00:00:09:00'),
            end=Timecode.from_hhmmssff('00:00:14:00'),
        )
        sampled_frames = [
            make_stage3_art_state_sample(285, changed=False),
            make_stage3_art_state_sample(290, changed=False),
            make_stage3_art_state_sample(366, changed=True),
            make_stage3_art_state_sample(370, changed=True),
        ]

        screened = screen_stage3_candidate_unions(
            [first_union, second_union],
            records,
            sampled_frames=sampled_frames,
            chapter_range=chapter_range,
            settings=make_settings(),
        )

        self.assertEqual(screened[0].screening_result, 'rejected')
        self.assertEqual(screened[0].reason, 'rejected_after_rescue_failure')
        self.assertEqual(screened[0].stage3_mode, 'snapshot_rescue')

# ============================================================
# SECTION F - Stage 4 Time Slice Classification Tests
# ============================================================


class TimeSliceClassificationTests(unittest.TestCase):
    def test_stage4_leading_zero_change_holding_probe_is_not_retained_when_followed_by_positive_probe(self) -> None:
        screened_union = make_screened_union(start_frame=300, end_frame=420)
        holding_probe = ClassifiedTimeSlice(
            slice_index=1,
            parent_union_index=1,
            slice_level=0,
            start_frame=300,
            end_frame=360,
            start_time='00:00:10:00',
            end_time='00:00:12:00',
            footprint=frozenset({(2, 2)}),
            footprint_size=1,
            within_slice_record_count=1,
            classification='undetermined',
            reason='mixed_slice_evidence',
            lasting_change_evidence_score=0.0,
            before_reference_activity=0.0,
            after_reference_activity=0.0,
            reference_windows_reliable=True,
        )
        positive_probe = replace(
            holding_probe,
            slice_index=2,
            start_frame=360,
            end_frame=420,
            start_time='00:00:12:00',
            end_time='00:00:14:00',
            classification='valid',
            reason='slice_activity_supported',
            lasting_change_evidence_score=0.8,
        )
        probe_results = [
            (holding_probe, {'probe_label': 'holding', 'changed_support_score': 0.0, 'judgeable_changed_support_score': 0.0, 'lasting_change_evidence_score': 0.0, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
            (positive_probe, {'probe_label': 'positive', 'changed_support_score': 0.30, 'lasting_change_evidence_score': 0.8, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
        ]

        bands = build_stage4_bands_from_probe_results(screened_union, probe_results)

        self.assertEqual(len(bands), 1)
        self.assertEqual((bands[0].start_frame, bands[0].end_frame), (360, 420))

    def test_stage4_changed_holding_probe_is_retained_when_followed_by_positive_probe(self) -> None:
        screened_union = make_screened_union(start_frame=300, end_frame=420)
        holding_probe = ClassifiedTimeSlice(
            slice_index=1,
            parent_union_index=1,
            slice_level=0,
            start_frame=300,
            end_frame=360,
            start_time='00:00:10:00',
            end_time='00:00:12:00',
            footprint=frozenset({(2, 2)}),
            footprint_size=1,
            within_slice_record_count=1,
            classification='undetermined',
            reason='mixed_slice_evidence',
            lasting_change_evidence_score=0.0,
            before_reference_activity=0.0,
            after_reference_activity=0.0,
            reference_windows_reliable=True,
        )
        positive_probe = replace(
            holding_probe,
            slice_index=2,
            start_frame=360,
            end_frame=420,
            start_time='00:00:12:00',
            end_time='00:00:14:00',
            classification='valid',
            reason='slice_activity_supported',
            lasting_change_evidence_score=0.8,
        )
        probe_results = [
            (holding_probe, {'probe_label': 'holding', 'changed_support_score': 0.03, 'judgeable_changed_support_score': 0.12, 'lasting_change_evidence_score': 0.0, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
            (positive_probe, {'probe_label': 'positive', 'changed_support_score': 0.30, 'judgeable_changed_support_score': 0.30, 'lasting_change_evidence_score': 0.8, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
        ]

        bands = build_stage4_bands_from_probe_results(screened_union, probe_results)

        self.assertEqual(len(bands), 1)
        self.assertEqual((bands[0].start_frame, bands[0].end_frame), (360, 420))

    def test_stage4_sustained_holding_probes_without_positive_support_do_not_open_band(self) -> None:
        screened_union = make_screened_union(start_frame=300, end_frame=480)
        first_probe = ClassifiedTimeSlice(
            slice_index=1,
            parent_union_index=1,
            slice_level=0,
            start_frame=300,
            end_frame=360,
            start_time='00:00:10:00',
            end_time='00:00:12:00',
            footprint=frozenset({(2, 2)}),
            footprint_size=1,
            within_slice_record_count=1,
            classification='undetermined',
            reason='mixed_slice_evidence',
            lasting_change_evidence_score=0.0,
            before_reference_activity=0.0,
            after_reference_activity=0.0,
            reference_windows_reliable=True,
        )
        second_probe = replace(first_probe, slice_index=2, start_frame=360, end_frame=420, start_time='00:00:12:00', end_time='00:00:14:00')
        negative_probe = replace(first_probe, slice_index=3, start_frame=420, end_frame=480, start_time='00:00:14:00', end_time='00:00:16:00', classification='invalid', reason='weak_slice_activity')
        probe_results = [
            (first_probe, {'probe_label': 'holding', 'changed_support_score': 0.0, 'judgeable_changed_support_score': 0.0, 'lasting_change_evidence_score': 0.0, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
            (second_probe, {'probe_label': 'holding', 'changed_support_score': 0.0, 'judgeable_changed_support_score': 0.0, 'lasting_change_evidence_score': 0.0, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
            (negative_probe, {'probe_label': 'negative', 'changed_support_score': 0.0, 'lasting_change_evidence_score': 0.0, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
        ]

        bands = build_stage4_bands_from_probe_results(screened_union, probe_results)

        self.assertEqual(bands, [])

    def test_stage4_structural_unclear_then_holding_can_backfill_opening_into_later_positive(self) -> None:
        screened_union = make_screened_union(start_frame=300, end_frame=540)
        unclear_probe = ClassifiedTimeSlice(
            slice_index=1,
            parent_union_index=1,
            slice_level=0,
            start_frame=355,
            end_frame=365,
            start_time='00:00:11:25',
            end_time='00:00:12:05',
            footprint=frozenset({(2, 2), (2, 3)}),
            footprint_size=2,
            within_slice_record_count=1,
            classification='undetermined',
            reason='probe_cell_ambiguous_support',
            lasting_change_evidence_score=0.0,
            before_reference_activity=0.0,
            after_reference_activity=0.0,
            reference_windows_reliable=True,
        )
        holding_probe = replace(
            unclear_probe,
            slice_index=2,
            start_frame=415,
            end_frame=425,
            start_time='00:00:13:25',
            end_time='00:00:14:05',
            reason='probe_cell_structural_holding_support',
        )
        positive_probe = replace(
            unclear_probe,
            slice_index=3,
            start_frame=475,
            end_frame=485,
            start_time='00:00:15:25',
            end_time='00:00:16:05',
            classification='valid',
            reason='probe_cell_changed_support',
            lasting_change_evidence_score=0.8,
        )
        probe_results = [
            (unclear_probe, {'probe_label': 'unclear', 'changed_support_score': 0.0, 'judgeable_changed_support_score': 0.0, 'confirmed_changed_cell_count': 0, 'unresolved_cell_count': 4, 'structural_holding_support': True, 'lasting_change_evidence_score': 0.0, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
            (holding_probe, {'probe_label': 'holding', 'changed_support_score': 0.0, 'judgeable_changed_support_score': 0.0, 'confirmed_changed_cell_count': 0, 'unresolved_cell_count': 6, 'structural_holding_support': True, 'lasting_change_evidence_score': 0.0, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
            (positive_probe, {'probe_label': 'positive', 'changed_support_score': 0.3, 'judgeable_changed_support_score': 0.3, 'confirmed_changed_cell_count': 2, 'unresolved_cell_count': 0, 'structural_holding_support': False, 'lasting_change_evidence_score': 0.8, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
        ]

        bands = build_stage4_bands_from_probe_results(screened_union, probe_results)

        self.assertEqual(len(bands), 1)
        self.assertEqual((bands[0].start_frame, bands[0].end_frame), (415, 485))

    def test_stage4_post_reset_contamination_expands_to_local_neighbors(self) -> None:
        contaminated = stage4_extract_post_reset_contaminated_cells({
            'probe_label': 'positive',
            'unresolved_cell_count': 5,
            'confirmed_changed_cell_count': 36,
            'changed_support_score': 0.62,
            'confirmed_changed_cells': [
                {'coordinate': [1, 1]},
                {'coordinate': [1, 4]},
                {'coordinate': [4, 1]},
                {'coordinate': [4, 4]},
            ],
            'unresolved_cells': [
                {'coordinate': [6, 9]},
                {'coordinate': [6, 10]},
                {'coordinate': [7, 10]},
                {'coordinate': [8, 9]},
                {'coordinate': [8, 10]},
            ],
            'recently_active_cells': [
                {'coordinate': [6, 9]},
                {'coordinate': [6, 10]},
                {'coordinate': [6, 11]},
                {'coordinate': [7, 10]},
                {'coordinate': [7, 11]},
                {'coordinate': [8, 9]},
                {'coordinate': [8, 10]},
            ],
        })

        self.assertEqual(
            contaminated,
            frozenset({(6, 9), (6, 10), (6, 11), (7, 10), (7, 11), (8, 9), (8, 10)}),
        )

    def test_stage4_near_global_opening_attribution_finds_earlier_broad_reset_frame(self) -> None:
        screened_union = make_screened_union(start_frame=7000, end_frame=7200)
        target_coordinates = frozenset(
            (row_index, column_index)
            for row_index in range(GRID_ROWS)
            for column_index in range(GRID_COLUMNS)
        )
        records = [
            make_record(1, 7008, touched_grid_coordinates=((6, 9), (6, 10), (7, 9))),
            make_record(2, 7017, touched_grid_coordinates=tuple(target_coordinates)),
            make_record(3, 7038, touched_grid_coordinates=((5, 10), (5, 11), (6, 11), (6, 10))),
            make_record(4, 7039, touched_grid_coordinates=((4, 10), (5, 10), (5, 11), (6, 11), (6, 10))),
        ]

        attribution_start = find_stage4_near_global_opening_attribution_start_frame(
            screened_candidate_union=screened_union,
            ordered_records=records,
            search_start_frame=7010,
            slice_end=7046,
            target_coordinates=target_coordinates,
        )

        self.assertEqual(attribution_start, 7017)
    def test_stage4_structural_unclear_bridge_keeps_active_band_alive(self) -> None:
        screened_union = make_screened_union(start_frame=300, end_frame=540)
        first_positive_probe = ClassifiedTimeSlice(
            slice_index=1,
            parent_union_index=1,
            slice_level=0,
            start_frame=355,
            end_frame=365,
            start_time='00:00:11:25',
            end_time='00:00:12:05',
            footprint=frozenset({(5, 5), (5, 6)}),
            footprint_size=2,
            within_slice_record_count=1,
            classification='valid',
            reason='probe_cell_changed_support',
            lasting_change_evidence_score=0.8,
            before_reference_activity=0.0,
            after_reference_activity=0.0,
            reference_windows_reliable=True,
        )
        ambiguous_bridge_probe = replace(
            first_positive_probe,
            slice_index=2,
            start_frame=415,
            end_frame=425,
            start_time='00:00:13:25',
            end_time='00:00:14:05',
            classification='undetermined',
            reason='probe_cell_ambiguous_support',
            lasting_change_evidence_score=0.0,
        )
        second_positive_probe = replace(
            first_positive_probe,
            slice_index=3,
            start_frame=475,
            end_frame=485,
            start_time='00:00:15:25',
            end_time='00:00:16:05',
            footprint=frozenset({(6, 6), (6, 7)}),
        )
        probe_results = [
            (first_positive_probe, {'probe_label': 'positive', 'changed_support_score': 0.30, 'judgeable_changed_support_score': 0.30, 'confirmed_changed_cell_count': 2, 'unresolved_cell_count': 0, 'structural_holding_support': False, 'lasting_change_evidence_score': 0.8, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
            (ambiguous_bridge_probe, {'probe_label': 'unclear', 'changed_support_score': 0.0, 'judgeable_changed_support_score': 0.0, 'confirmed_changed_cell_count': 0, 'unresolved_cell_count': 2, 'structural_holding_support': True, 'lasting_change_evidence_score': 0.0, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
            (second_positive_probe, {'probe_label': 'positive', 'changed_support_score': 0.30, 'judgeable_changed_support_score': 0.30, 'confirmed_changed_cell_count': 2, 'unresolved_cell_count': 0, 'structural_holding_support': False, 'lasting_change_evidence_score': 0.8, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
        ]

        bands = build_stage4_bands_from_probe_results(screened_union, probe_results)

        self.assertEqual(len(bands), 1)
        self.assertEqual((bands[0].start_frame, bands[0].end_frame), (355, 485))

    def test_stage4_two_positive_chain_allows_single_unresolved_unclear_bridge(self) -> None:
        screened_union = make_screened_union(start_frame=300, end_frame=600)
        first_positive_probe = ClassifiedTimeSlice(
            slice_index=1,
            parent_union_index=1,
            slice_level=0,
            start_frame=355,
            end_frame=365,
            start_time='00:00:11:25',
            end_time='00:00:12:05',
            footprint=frozenset({(5, 5), (5, 6)}),
            footprint_size=2,
            within_slice_record_count=1,
            classification='valid',
            reason='probe_cell_changed_support',
            lasting_change_evidence_score=0.8,
            before_reference_activity=0.0,
            after_reference_activity=0.0,
            reference_windows_reliable=True,
        )
        second_positive_probe = replace(
            first_positive_probe,
            slice_index=2,
            start_frame=415,
            end_frame=425,
            start_time='00:00:13:25',
            end_time='00:00:14:05',
        )
        ambiguous_bridge_probe = replace(
            first_positive_probe,
            slice_index=3,
            start_frame=475,
            end_frame=485,
            start_time='00:00:15:25',
            end_time='00:00:16:05',
            classification='undetermined',
            reason='probe_cell_ambiguous_support',
            lasting_change_evidence_score=0.0,
        )
        third_positive_probe = replace(
            first_positive_probe,
            slice_index=4,
            start_frame=535,
            end_frame=545,
            start_time='00:00:17:25',
            end_time='00:00:18:05',
            footprint=frozenset({(6, 6), (6, 7)}),
        )
        probe_results = [
            (first_positive_probe, {'probe_label': 'positive', 'changed_support_score': 0.30, 'judgeable_changed_support_score': 0.30, 'confirmed_changed_cell_count': 2, 'unresolved_cell_count': 0, 'structural_holding_support': False, 'lasting_change_evidence_score': 0.8, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
            (second_positive_probe, {'probe_label': 'positive', 'changed_support_score': 0.12, 'judgeable_changed_support_score': 0.12, 'confirmed_changed_cell_count': 1, 'unresolved_cell_count': 2, 'structural_holding_support': True, 'lasting_change_evidence_score': 0.4, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
            (ambiguous_bridge_probe, {'probe_label': 'unclear', 'changed_support_score': 0.0, 'judgeable_changed_support_score': 0.0, 'confirmed_changed_cell_count': 0, 'unresolved_cell_count': 1, 'structural_holding_support': True, 'lasting_change_evidence_score': 0.0, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
            (third_positive_probe, {'probe_label': 'positive', 'changed_support_score': 0.30, 'judgeable_changed_support_score': 0.30, 'confirmed_changed_cell_count': 2, 'unresolved_cell_count': 0, 'structural_holding_support': False, 'lasting_change_evidence_score': 0.8, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
        ]

        bands = build_stage4_bands_from_probe_results(screened_union, probe_results)

        self.assertEqual(len(bands), 1)
        self.assertEqual((bands[0].start_frame, bands[0].end_frame), (355, 545))

    def test_stage4_single_unresolved_unclear_bridge_does_not_keep_active_band_alive(self) -> None:
        screened_union = make_screened_union(start_frame=300, end_frame=540)
        first_positive_probe = ClassifiedTimeSlice(
            slice_index=1,
            parent_union_index=1,
            slice_level=0,
            start_frame=355,
            end_frame=365,
            start_time='00:00:11:25',
            end_time='00:00:12:05',
            footprint=frozenset({(5, 5), (5, 6)}),
            footprint_size=2,
            within_slice_record_count=1,
            classification='valid',
            reason='probe_cell_changed_support',
            lasting_change_evidence_score=0.8,
            before_reference_activity=0.0,
            after_reference_activity=0.0,
            reference_windows_reliable=True,
        )
        ambiguous_bridge_probe = replace(
            first_positive_probe,
            slice_index=2,
            start_frame=415,
            end_frame=425,
            start_time='00:00:13:25',
            end_time='00:00:14:05',
            classification='undetermined',
            reason='probe_cell_ambiguous_support',
            lasting_change_evidence_score=0.0,
        )
        second_positive_probe = replace(
            first_positive_probe,
            slice_index=3,
            start_frame=475,
            end_frame=485,
            start_time='00:00:15:25',
            end_time='00:00:16:05',
            footprint=frozenset({(6, 6), (6, 7)}),
        )
        probe_results = [
            (first_positive_probe, {'probe_label': 'positive', 'changed_support_score': 0.30, 'judgeable_changed_support_score': 0.30, 'confirmed_changed_cell_count': 2, 'unresolved_cell_count': 0, 'structural_holding_support': False, 'lasting_change_evidence_score': 0.8, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
            (ambiguous_bridge_probe, {'probe_label': 'unclear', 'changed_support_score': 0.0, 'judgeable_changed_support_score': 0.0, 'confirmed_changed_cell_count': 0, 'unresolved_cell_count': 1, 'structural_holding_support': True, 'lasting_change_evidence_score': 0.0, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
            (second_positive_probe, {'probe_label': 'positive', 'changed_support_score': 0.30, 'judgeable_changed_support_score': 0.30, 'confirmed_changed_cell_count': 2, 'unresolved_cell_count': 0, 'structural_holding_support': False, 'lasting_change_evidence_score': 0.8, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
        ]

        bands = build_stage4_bands_from_probe_results(screened_union, probe_results)

        self.assertEqual(len(bands), 2)
        self.assertEqual([(band.start_frame, band.end_frame) for band in bands], [(355, 365), (475, 485)])

    def test_stage4_late_resolution_probe_closes_earlier_band_and_opens_from_union_start(self) -> None:
        screened_union = make_screened_union(start_frame=300, end_frame=540)
        opening_probe = ClassifiedTimeSlice(
            slice_index=1,
            parent_union_index=1,
            slice_level=0,
            start_frame=355,
            end_frame=365,
            start_time='00:00:11:25',
            end_time='00:00:12:05',
            footprint=frozenset({(2, 2), (2, 3)}),
            footprint_size=2,
            within_slice_record_count=1,
            classification='undetermined',
            reason='probe_cell_ambiguous_support',
            lasting_change_evidence_score=0.0,
            before_reference_activity=0.0,
            after_reference_activity=0.0,
            reference_windows_reliable=True,
        )
        holding_probe = replace(
            opening_probe,
            slice_index=2,
            start_frame=415,
            end_frame=425,
            start_time='00:00:13:25',
            end_time='00:00:14:05',
            reason='probe_cell_structural_holding_support',
        )
        late_resolution_probe = replace(
            opening_probe,
            slice_index=3,
            start_frame=475,
            end_frame=485,
            start_time='00:00:15:25',
            end_time='00:00:16:05',
            reason='probe_cell_late_resolution_support',
        )
        later_positive_probe = replace(
            opening_probe,
            slice_index=4,
            start_frame=535,
            end_frame=540,
            start_time='00:00:17:25',
            end_time='00:00:18:00',
            classification='valid',
            reason='probe_cell_changed_support',
            lasting_change_evidence_score=0.8,
        )
        probe_results = [
            (opening_probe, {'probe_label': 'unclear', 'changed_support_score': 0.0, 'judgeable_changed_support_score': 0.0, 'confirmed_changed_cell_count': 0, 'unresolved_cell_count': 4, 'structural_holding_support': True, 'opening_zone_low_confidence': True, 'late_resolution_only': False, 'lasting_change_evidence_score': 0.0, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
            (holding_probe, {'probe_label': 'holding', 'changed_support_score': 0.0, 'judgeable_changed_support_score': 0.0, 'confirmed_changed_cell_count': 0, 'unresolved_cell_count': 6, 'structural_holding_support': True, 'opening_zone_low_confidence': False, 'late_resolution_only': False, 'lasting_change_evidence_score': 0.0, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
            (late_resolution_probe, {'probe_label': 'holding', 'changed_support_score': 0.1, 'judgeable_changed_support_score': 0.2, 'confirmed_changed_cell_count': 2, 'unresolved_cell_count': 0, 'structural_holding_support': False, 'opening_zone_low_confidence': False, 'late_resolution_only': True, 'lasting_change_evidence_score': 0.1, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
            (later_positive_probe, {'probe_label': 'positive', 'changed_support_score': 0.3, 'judgeable_changed_support_score': 0.3, 'confirmed_changed_cell_count': 2, 'unresolved_cell_count': 0, 'structural_holding_support': False, 'opening_zone_low_confidence': False, 'late_resolution_only': False, 'lasting_change_evidence_score': 0.8, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
        ]

        bands = build_stage4_bands_from_probe_results(screened_union, probe_results)

        self.assertEqual(len(bands), 2)
        self.assertEqual([(band.start_frame, band.end_frame) for band in bands], [(415, 425), (535, 540)])

    def test_stage4_first_positive_probe_can_open_from_local_attribution_start(self) -> None:
        screened_union = make_screened_union(start_frame=300, end_frame=540)
        positive_probe = ClassifiedTimeSlice(
            slice_index=1,
            parent_union_index=1,
            slice_level=0,
            start_frame=355,
            end_frame=365,
            start_time='00:00:11:25',
            end_time='00:00:12:05',
            footprint=frozenset({(6, 8), (7, 8)}),
            footprint_size=2,
            within_slice_record_count=1,
            classification='valid',
            reason='probe_cell_changed_support',
            lasting_change_evidence_score=0.8,
            before_reference_activity=0.0,
            after_reference_activity=0.0,
            reference_windows_reliable=True,
        )
        probe_results = [
            (positive_probe, {'probe_label': 'positive', 'changed_support_score': 0.30, 'judgeable_changed_support_score': 0.30, 'confirmed_changed_cell_count': 2, 'unresolved_cell_count': 0, 'structural_holding_support': False, 'opening_zone_low_confidence': True, 'opening_attribution_start_frame': 320, 'opening_attribution_start_time': '00:00:10:20', 'late_resolution_only': False, 'lasting_change_evidence_score': 0.8, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
        ]

        bands = build_stage4_bands_from_probe_results(screened_union, probe_results)

        self.assertEqual(len(bands), 1)
        self.assertEqual((bands[0].start_frame, bands[0].end_frame), (320, 365))


    def test_stage4_generalized_opening_attribution_can_open_from_recent_clean_break(self) -> None:
        screened_union = make_screened_union(start_frame=300, end_frame=540)
        positive_probe = ClassifiedTimeSlice(
            slice_index=1,
            parent_union_index=1,
            slice_level=0,
            start_frame=355,
            end_frame=365,
            start_time='00:00:11:25',
            end_time='00:00:12:05',
            footprint=frozenset({(6, 8), (7, 8)}),
            footprint_size=2,
            within_slice_record_count=1,
            classification='valid',
            reason='probe_cell_changed_support',
            lasting_change_evidence_score=0.8,
            before_reference_activity=0.0,
            after_reference_activity=0.0,
            reference_windows_reliable=True,
        )
        probe_results = [
            (positive_probe, {'probe_label': 'positive', 'changed_support_score': 0.30, 'judgeable_changed_support_score': 0.30, 'confirmed_changed_cell_count': 2, 'unresolved_cell_count': 0, 'structural_holding_support': False, 'opening_zone_low_confidence': False, 'opening_attribution_start_frame': 320, 'opening_attribution_start_time': '00:00:10:20', 'late_resolution_only': False, 'lasting_change_evidence_score': 0.8, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
        ]

        bands = build_stage4_bands_from_probe_results(screened_union, probe_results)

        self.assertEqual(len(bands), 1)
        self.assertEqual((bands[0].start_frame, bands[0].end_frame), (320, 365))

    def test_stage4_generalized_opening_attribution_does_not_reach_too_far_back(self) -> None:
        screened_union = make_screened_union(start_frame=300, end_frame=720)
        positive_probe = ClassifiedTimeSlice(
            slice_index=1,
            parent_union_index=1,
            slice_level=0,
            start_frame=535,
            end_frame=545,
            start_time='00:00:17:25',
            end_time='00:00:18:05',
            footprint=frozenset({(6, 8), (7, 8)}),
            footprint_size=2,
            within_slice_record_count=1,
            classification='valid',
            reason='probe_cell_changed_support',
            lasting_change_evidence_score=0.8,
            before_reference_activity=0.0,
            after_reference_activity=0.0,
            reference_windows_reliable=True,
        )
        probe_results = [
            (positive_probe, {'probe_label': 'positive', 'changed_support_score': 0.08, 'judgeable_changed_support_score': 0.08, 'confirmed_changed_cell_count': 2, 'unresolved_cell_count': 1, 'structural_holding_support': True, 'opening_zone_low_confidence': False, 'opening_attribution_start_frame': 355, 'opening_attribution_start_time': '00:00:11:25', 'late_resolution_only': False, 'lasting_change_evidence_score': 0.08, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
        ]

        bands = build_stage4_bands_from_probe_results(screened_union, probe_results)

        self.assertEqual(len(bands), 1)
        self.assertEqual((bands[0].start_frame, bands[0].end_frame), (535, 545))
    def test_stage4_single_front_loaded_touch_becomes_late_resolution_holding(self) -> None:
        screened_union = make_screened_union(
            start_frame=3900,
            end_frame=4200,
            union_footprint=frozenset({(3, 2), (3, 3), (4, 1), (4, 2), (4, 3), (5, 1), (5, 2), (5, 3), (6, 2)}),
        )
        records = [
            make_record(
                1,
                3967,
                movement_present=True,
                movement_strength_score=0.7,
                temporal_persistence_score=0.6,
                spatial_extent_score=0.4,
                touched_grid_coordinates=((3, 2), (3, 3), (4, 2), (4, 3), (5, 2), (5, 3)),
            ),
        ]
        sampled_frames = [
            make_stage3_art_state_sample(3890, changed=False),
            make_stage3_art_state_sample(3894, changed=False),
            make_stage3_art_state_sample(3972, changed_cells=((4, 2), (5, 2))),
            make_stage3_art_state_sample(3974, changed_cells=((4, 2), (5, 2))),
            make_stage3_art_state_sample(3980, changed_cells=((4, 2), (5, 2))),
        ]

        evaluation = evaluate_stage4_cell_level_probe(
            screened_candidate_union=screened_union,
            slice_start=3967,
            slice_end=3977,
            ordered_records=records,
            sampled_frames=sampled_frames,
            settings=make_settings(),
            carried_unresolved_cells=frozenset({(4, 2), (5, 2)}),
            carried_recently_changed_cells=frozenset(),
            baseline_comparison_state={},
            previous_probe_end=3957,
        )

        self.assertTrue(evaluation['late_resolution_only'])
        self.assertEqual(evaluation['probe_label'], 'holding')
        self.assertEqual(evaluation['confirmed_changed_cell_count'], 2)
        self.assertEqual(evaluation['changed_touch_frame_count'], 1)
        self.assertLessEqual(evaluation['changed_touch_frame_ratio'], 0.10)
    def test_stage4_previous_broad_negative_blocks_weak_opening_attribution(self) -> None:
        screened_union = make_screened_union(
            start_frame=300,
            end_frame=600,
            union_footprint=frozenset({(6, 8), (6, 9), (7, 8)}),
        )
        records = [
            make_record(
                1,
                535,
                movement_present=True,
                movement_strength_score=0.75,
                temporal_persistence_score=0.6,
                spatial_extent_score=0.4,
                touched_grid_coordinates=((6, 8),),
            ),
            make_record(
                2,
                539,
                movement_present=True,
                movement_strength_score=0.72,
                temporal_persistence_score=0.58,
                spatial_extent_score=0.4,
                touched_grid_coordinates=((7, 8),),
            ),
        ]
        sampled_frames = [
            make_stage3_art_state_sample(500, changed=False),
            make_stage3_art_state_sample(504, changed=False),
            make_stage3_art_state_sample(540, changed_cells=((6, 8), (7, 8))),
            make_stage3_art_state_sample(544, changed_cells=((6, 8), (7, 8))),
            make_stage3_art_state_sample(548, changed_cells=((6, 8), (7, 8))),
        ]

        evaluation = evaluate_stage4_cell_level_probe(
            screened_candidate_union=screened_union,
            slice_start=535,
            slice_end=545,
            ordered_records=records,
            sampled_frames=sampled_frames,
            settings=make_settings(),
            carried_unresolved_cells=frozenset(),
            carried_recently_changed_cells=frozenset(),
            carried_post_reset_contaminated_cells=frozenset(),
            baseline_comparison_state={},
            previous_probe_end=525,
            previous_probe_blocks_opening_attribution=True,
        )

        self.assertEqual(evaluation['probe_label'], 'positive')
        self.assertIsNone(evaluation['opening_attribution_start_frame'])
    def test_stage4_changed_frontier_opening_attribution_can_cross_previous_broad_negative(self) -> None:
        screened_union = make_screened_union(
            start_frame=300,
            end_frame=600,
            union_footprint=frozenset({(5, 2), (5, 3), (6, 2), (6, 3), (7, 2), (7, 3)}),
        )
        records = [
            make_record(
                1,
                530,
                movement_present=True,
                movement_strength_score=0.78,
                temporal_persistence_score=0.62,
                spatial_extent_score=0.4,
                touched_grid_coordinates=((5, 2), (6, 2), (7, 2)),
            ),
            make_record(
                2,
                533,
                movement_present=True,
                movement_strength_score=0.80,
                temporal_persistence_score=0.64,
                spatial_extent_score=0.42,
                touched_grid_coordinates=((5, 2), (5, 3), (6, 2), (6, 3)),
            ),
            make_record(
                3,
                537,
                movement_present=True,
                movement_strength_score=0.82,
                temporal_persistence_score=0.66,
                spatial_extent_score=0.45,
                touched_grid_coordinates=((5, 2), (5, 3), (6, 2), (6, 3), (7, 2)),
            ),
        ]
        sampled_frames = [
            make_stage3_art_state_sample(500, changed=False),
            make_stage3_art_state_sample(504, changed=False),
            make_stage3_art_state_sample(540, changed_cells=((5, 2), (5, 3), (6, 2), (6, 3), (7, 2), (7, 3))),
            make_stage3_art_state_sample(544, changed_cells=((5, 2), (5, 3), (6, 2), (6, 3), (7, 2), (7, 3))),
            make_stage3_art_state_sample(548, changed_cells=((5, 2), (5, 3), (6, 2), (6, 3), (7, 2), (7, 3))),
        ]

        evaluation = evaluate_stage4_cell_level_probe(
            screened_candidate_union=screened_union,
            slice_start=535,
            slice_end=545,
            ordered_records=records,
            sampled_frames=sampled_frames,
            settings=make_settings(),
            carried_unresolved_cells=frozenset(),
            carried_recently_changed_cells=frozenset(),
            carried_post_reset_contaminated_cells=frozenset(),
            baseline_comparison_state={},
            previous_probe_end=525,
            previous_probe_blocks_opening_attribution=True,
        )

        self.assertEqual(evaluation['probe_label'], 'positive')
        self.assertEqual(evaluation['opening_attribution_start_frame'], 530)
    def test_stage4_builds_separate_valid_bands_when_a_negative_probe_gap_splits_union_activity(self) -> None:
        screened_union = make_screened_union(start_frame=300, end_frame=540)
        first_positive_probe = ClassifiedTimeSlice(
            slice_index=1,
            parent_union_index=1,
            slice_level=0,
            start_frame=355,
            end_frame=365,
            start_time='00:00:11:25',
            end_time='00:00:12:05',
            footprint=frozenset({(5, 5), (5, 6), (6, 5), (6, 6)}),
            footprint_size=4,
            within_slice_record_count=1,
            classification='valid',
            reason='probe_cell_changed_support',
            lasting_change_evidence_score=0.8,
            before_reference_activity=0.0,
            after_reference_activity=0.0,
            reference_windows_reliable=True,
        )
        negative_probe = replace(
            first_positive_probe,
            slice_index=2,
            start_frame=415,
            end_frame=425,
            start_time='00:00:13:25',
            end_time='00:00:14:05',
            classification='invalid',
            reason='probe_cell_broad_unchanged_support',
            lasting_change_evidence_score=0.0,
        )
        second_positive_probe = replace(
            first_positive_probe,
            slice_index=3,
            start_frame=475,
            end_frame=485,
            start_time='00:00:15:25',
            end_time='00:00:16:05',
            footprint=frozenset({(2, 8), (2, 9), (3, 8), (3, 9)}),
        )
        probe_results = [
            (first_positive_probe, {'probe_label': 'positive', 'changed_support_score': 0.30, 'judgeable_changed_support_score': 0.30, 'lasting_change_evidence_score': 0.8, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
            (negative_probe, {'probe_label': 'negative', 'changed_support_score': 0.0, 'judgeable_changed_support_score': 0.0, 'lasting_change_evidence_score': 0.0, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
            (second_positive_probe, {'probe_label': 'positive', 'changed_support_score': 0.30, 'judgeable_changed_support_score': 0.30, 'lasting_change_evidence_score': 0.8, 'before_reference_activity': 0.0, 'after_reference_activity': 0.0, 'reference_windows_reliable': True}),
        ]

        bands = build_stage4_bands_from_probe_results(screened_union, probe_results)

        self.assertEqual(len(bands), 2)
        self.assertEqual([(band.start_frame, band.end_frame) for band in bands], [(355, 365), (475, 485)])
        self.assertTrue(all(band.classification == 'valid' for band in bands))
        self.assertTrue(all(band.reason == 'probe_supported_band' for band in bands))

    def test_stage4_merges_consecutive_positive_probes_into_one_valid_band(self) -> None:
        screened_union = make_screened_union(
            start_frame=300,
            end_frame=420,
            union_footprint=frozenset({(5, 5), (5, 6), (6, 5), (6, 6), (2, 8), (2, 9), (3, 8), (3, 9)}),
        )
        records = [
            make_record(1, 304, movement_present=True, movement_strength_score=0.86, temporal_persistence_score=0.84, spatial_extent_score=0.88, touched_grid_coordinates=((5, 5), (5, 6), (6, 5), (6, 6))),
            make_record(2, 310, movement_present=True, movement_strength_score=0.85, temporal_persistence_score=0.83, spatial_extent_score=0.87, touched_grid_coordinates=((5, 5), (5, 6), (6, 5), (6, 6))),
            make_record(3, 404, movement_present=True, movement_strength_score=0.86, temporal_persistence_score=0.84, spatial_extent_score=0.88, touched_grid_coordinates=((2, 8), (2, 9), (3, 8), (3, 9))),
            make_record(4, 408, movement_present=True, movement_strength_score=0.85, temporal_persistence_score=0.83, spatial_extent_score=0.87, touched_grid_coordinates=((2, 8), (2, 9), (3, 8), (3, 9))),
            make_record(5, 414, movement_present=True, movement_strength_score=0.82, temporal_persistence_score=0.80, spatial_extent_score=0.40, touched_grid_coordinates=((0, 0), (0, 1))),
        ]
        sampled_frames = [
            make_stage3_art_state_sample(288, changed=False),
            make_stage3_art_state_sample(294, changed=False),
            make_stage3_art_state_sample(348, changed=False),
            make_stage3_art_state_sample(354, changed=False),
            make_stage3_art_state_sample(366, changed_cells=((5, 5), (5, 6), (6, 5), (6, 6))),
            make_stage3_art_state_sample(368, changed_cells=((5, 5), (5, 6), (6, 5), (6, 6))),
            make_stage3_art_state_sample(372, changed_cells=((5, 5), (5, 6), (6, 5), (6, 6))),
            make_stage3_art_state_sample(411, changed_cells=((5, 5), (5, 6), (6, 5), (6, 6), (2, 8), (2, 9), (3, 8), (3, 9))),
            make_stage3_art_state_sample(413, changed_cells=((5, 5), (5, 6), (6, 5), (6, 6), (2, 8), (2, 9), (3, 8), (3, 9))),
            make_stage3_art_state_sample(415, changed_cells=((5, 5), (5, 6), (6, 5), (6, 6), (2, 8), (2, 9), (3, 8), (3, 9))),
            make_stage3_art_state_sample(417, changed_cells=((5, 5), (5, 6), (6, 5), (6, 6), (2, 8), (2, 9), (3, 8), (3, 9))),
            make_stage3_art_state_sample(419, changed_cells=((5, 5), (5, 6), (6, 5), (6, 6), (2, 8), (2, 9), (3, 8), (3, 9))),
        ]

        slices = classify_stage4_time_slices([screened_union], records, sampled_frames=sampled_frames, settings=make_settings())

        self.assertEqual(len(slices), 1)
        self.assertEqual((slices[0].start_frame, slices[0].end_frame), (355, 420))
        self.assertEqual(slices[0].classification, 'valid')
        self.assertEqual(slices[0].reason, 'probe_supported_band')
    def test_stage4_ignores_rejected_unions(self) -> None:
        rejected_union = make_screened_union(surviving=False, screening_result='rejected')
        records = [
            make_record(1, 304, movement_present=True, movement_strength_score=0.7, temporal_persistence_score=0.65),
            make_record(2, 316, movement_present=True, movement_strength_score=0.68, temporal_persistence_score=0.62),
        ]

        slices = classify_stage4_time_slices([rejected_union], records, sampled_frames=[], settings=make_settings())
        self.assertEqual(slices, [])


# ============================================================
# SECTION G - Stage 5 Recursive Sub-Slice Refinement Tests
# ============================================================


class SubSliceRefinementTests(unittest.TestCase):
    def test_stage5_returns_stage4_bands_without_recursive_subdivision_when_references_are_unavailable(self) -> None:
        first_band = ClassifiedTimeSlice(
            slice_index=1,
            parent_union_index=1,
            slice_level=0,
            start_frame=300,
            end_frame=360,
            start_time='00:00:10:00',
            end_time='00:00:12:00',
            footprint=frozenset({(2, 2), (2, 3)}),
            footprint_size=2,
            within_slice_record_count=2,
            classification='valid',
            reason='probe_supported_band',
            lasting_change_evidence_score=0.72,
            before_reference_activity=0.02,
            after_reference_activity=0.02,
            reference_windows_reliable=True,
        )
        second_band = ClassifiedTimeSlice(
            slice_index=2,
            parent_union_index=1,
            slice_level=0,
            start_frame=420,
            end_frame=480,
            start_time='00:00:14:00',
            end_time='00:00:16:00',
            footprint=frozenset({(2, 4), (2, 5)}),
            footprint_size=2,
            within_slice_record_count=2,
            classification='valid',
            reason='probe_supported_band',
            lasting_change_evidence_score=0.70,
            before_reference_activity=0.02,
            after_reference_activity=0.02,
            reference_windows_reliable=True,
        )

        refined = refine_stage5_sub_slices([], [second_band, first_band], [], sampled_frames=[], settings=make_settings(), minimum_subdivision_frames=15)

        self.assertEqual([(slice_info.start_frame, slice_info.end_frame) for slice_info in refined], [(300, 360), (420, 480)])
        self.assertEqual([slice_info.reason for slice_info in refined], ['probe_supported_band', 'probe_supported_band'])

    def test_stage5_trims_valid_slice_start_using_three_frame_confirmation_window(self) -> None:
        screened_union = make_screened_union(
            start_frame=80,
            end_frame=180,
            union_footprint=frozenset({(2, 2)}),
        )
        valid_slice = ClassifiedTimeSlice(
            slice_index=1,
            parent_union_index=screened_union.candidate_union.union_index,
            slice_level=0,
            start_frame=100,
            end_frame=140,
            start_time='00:00:03:10',
            end_time='00:00:04:20',
            footprint=frozenset({(2, 2)}),
            footprint_size=1,
            within_slice_record_count=3,
            classification='valid',
            reason='probe_supported_band',
            lasting_change_evidence_score=0.8,
            before_reference_activity=0.0,
            after_reference_activity=0.0,
            reference_windows_reliable=True,
        )
        records = [
            make_record(1, 112, movement_present=True, touched_grid_coordinates=((2, 2),)),
            make_record(2, 113, movement_present=True, touched_grid_coordinates=((2, 2),)),
            make_record(3, 114, movement_present=True, touched_grid_coordinates=((2, 2),)),
        ]
        sampled_frames = [
            make_stage3_art_state_sample(88, changed=False),
            make_stage3_art_state_sample(90, changed=False),
            make_stage3_art_state_sample(92, changed=False),
            make_stage3_art_state_sample(94, changed=False),
            make_stage3_art_state_sample(96, changed=False),
            make_stage3_art_state_sample(98, changed=False),
            make_stage3_art_state_sample(108, changed=False),
            make_stage3_art_state_sample(109, changed=False),
            make_stage3_art_state_sample(110, changed=False),
            make_stage3_art_state_sample(112, changed_cells=((2, 2),)),
            make_stage3_art_state_sample(113, changed_cells=((2, 2),)),
            make_stage3_art_state_sample(114, changed_cells=((2, 2),)),
            make_stage3_art_state_sample(118, changed_cells=((2, 2),)),
        ]

        refined = refine_stage5_sub_slices(
            [screened_union],
            [valid_slice],
            records,
            sampled_frames=sampled_frames,
            settings=make_settings(),
            minimum_subdivision_frames=15,
        )

        self.assertEqual(len(refined), 1)
        self.assertEqual(refined[0].start_frame, 111)
        self.assertEqual(refined[0].end_frame, 140)

    def test_stage5_trims_valid_slice_end_using_latest_changed_cell_touch(self) -> None:
        screened_union = make_screened_union(
            start_frame=80,
            end_frame=180,
            union_footprint=frozenset({(2, 2)}),
        )
        valid_slice = ClassifiedTimeSlice(
            slice_index=1,
            parent_union_index=screened_union.candidate_union.union_index,
            slice_level=0,
            start_frame=100,
            end_frame=140,
            start_time='00:00:03:10',
            end_time='00:00:04:20',
            footprint=frozenset({(2, 2)}),
            footprint_size=1,
            within_slice_record_count=3,
            classification='valid',
            reason='probe_supported_band',
            lasting_change_evidence_score=0.8,
            before_reference_activity=0.0,
            after_reference_activity=0.0,
            reference_windows_reliable=True,
        )
        records = [
            make_record(1, 122, movement_present=True, touched_grid_coordinates=((2, 2),)),
            make_record(2, 125, movement_present=True, touched_grid_coordinates=((2, 2),)),
            make_record(3, 126, movement_present=True, touched_grid_coordinates=((2, 2),)),
        ]
        sampled_frames = [
            make_stage3_art_state_sample(88, changed=False),
            make_stage3_art_state_sample(90, changed=False),
            make_stage3_art_state_sample(92, changed=False),
            make_stage3_art_state_sample(94, changed=False),
            make_stage3_art_state_sample(96, changed=False),
            make_stage3_art_state_sample(98, changed=False),
            make_stage3_art_state_sample(122, changed_cells=((2, 2),)),
            make_stage3_art_state_sample(125, changed_cells=((2, 2),)),
            make_stage3_art_state_sample(126, changed_cells=((2, 2),)),
            make_stage3_art_state_sample(130, changed_cells=((2, 2),)),
        ]

        refined = refine_stage5_sub_slices(
            [screened_union],
            [valid_slice],
            records,
            sampled_frames=sampled_frames,
            settings=make_settings(),
            minimum_subdivision_frames=15,
        )

        self.assertEqual(len(refined), 1)
        self.assertEqual(refined[0].start_frame, 100)
        self.assertEqual(refined[0].end_frame, 127)
    def test_stage5_uses_broad_movement_start_for_large_footprint_slice(self) -> None:
        footprint = frozenset((row_index, column_index) for row_index in range(6) for column_index in range(6))
        screened_union = make_screened_union(
            start_frame=80,
            end_frame=220,
            union_footprint=footprint,
        )
        valid_slice = ClassifiedTimeSlice(
            slice_index=1,
            parent_union_index=screened_union.candidate_union.union_index,
            slice_level=0,
            start_frame=100,
            end_frame=180,
            start_time='00:00:03:10',
            end_time='00:00:06:00',
            footprint=footprint,
            footprint_size=len(footprint),
            within_slice_record_count=10,
            classification='valid',
            reason='probe_supported_band',
            lasting_change_evidence_score=0.8,
            before_reference_activity=0.0,
            after_reference_activity=0.0,
            reference_windows_reliable=True,
        )
        broad_touch = tuple((row_index, column_index) for row_index in range(6) for column_index in range(6))
        records = [
            make_record(1, 112, movement_present=True, touched_grid_coordinates=broad_touch),
            make_record(2, 113, movement_present=True, touched_grid_coordinates=broad_touch),
            make_record(3, 114, movement_present=True, touched_grid_coordinates=broad_touch),
        ]
        sampled_frames = [
            make_stage3_art_state_sample(88, changed=False),
            make_stage3_art_state_sample(90, changed=False),
            make_stage3_art_state_sample(92, changed=False),
            make_stage3_art_state_sample(94, changed=False),
            make_stage3_art_state_sample(96, changed=False),
            make_stage3_art_state_sample(98, changed=False),
            make_stage3_art_state_sample(112, changed_cells=broad_touch),
            make_stage3_art_state_sample(113, changed_cells=broad_touch),
            make_stage3_art_state_sample(114, changed_cells=broad_touch),
            make_stage3_art_state_sample(118, changed_cells=broad_touch),
        ]

        refined = refine_stage5_sub_slices(
            [screened_union],
            [valid_slice],
            records,
            sampled_frames=sampled_frames,
            settings=make_settings(),
            minimum_subdivision_frames=15,
        )

        self.assertEqual(len(refined), 1)
        self.assertEqual(refined[0].start_frame, 112)
        self.assertIsNotNone(refined[0].stage5_debug)
        self.assertEqual(refined[0].stage5_debug['start_broad_movement_frame'], 112)
        self.assertEqual(refined[0].stage5_debug['start_winning_path'], 'broad_movement')

    def test_stage5_broad_movement_requires_multi_frame_run(self) -> None:
        footprint = frozenset((row_index, column_index) for row_index in range(6) for column_index in range(6))
        broad_touch = tuple((row_index, column_index) for row_index in range(6) for column_index in range(6))
        valid_slice = ClassifiedTimeSlice(
            slice_index=1,
            parent_union_index=1,
            slice_level=0,
            start_frame=100,
            end_frame=180,
            start_time='00:00:03:10',
            end_time='00:00:06:00',
            footprint=footprint,
            footprint_size=len(footprint),
            within_slice_record_count=10,
            classification='valid',
            reason='probe_supported_band',
            lasting_change_evidence_score=0.8,
            before_reference_activity=0.0,
            after_reference_activity=0.0,
            reference_windows_reliable=True,
        )
        one_frame_records = [
            make_record(1, 112, movement_present=True, touched_grid_coordinates=broad_touch),
        ]
        three_frame_records = [
            make_record(1, 112, movement_present=True, touched_grid_coordinates=broad_touch),
            make_record(2, 113, movement_present=True, touched_grid_coordinates=broad_touch),
            make_record(3, 114, movement_present=True, touched_grid_coordinates=broad_touch),
        ]

        self.assertIsNone(
            find_stage5_broad_movement_boundary_frame(
                valid_slice,
                one_frame_records,
                search_start=100,
                search_end=140,
                prefer_latest=False,
            )
        )
        self.assertEqual(
            find_stage5_broad_movement_boundary_frame(
                valid_slice,
                three_frame_records,
                search_start=100,
                search_end=140,
                prefer_latest=False,
            ),
            112,
        )

    def test_stage5_vetoes_end_trim_that_would_violate_minimum_clip_length(self) -> None:
        footprint = frozenset((row_index, column_index) for row_index in range(6) for column_index in range(6))
        screened_union = make_screened_union(
            start_frame=80,
            end_frame=220,
            union_footprint=footprint,
        )
        valid_slice = ClassifiedTimeSlice(
            slice_index=1,
            parent_union_index=screened_union.candidate_union.union_index,
            slice_level=0,
            start_frame=100,
            end_frame=120,
            start_time='00:00:03:10',
            end_time='00:00:04:00',
            footprint=footprint,
            footprint_size=len(footprint),
            within_slice_record_count=10,
            classification='valid',
            reason='probe_supported_band',
            lasting_change_evidence_score=0.8,
            before_reference_activity=0.0,
            after_reference_activity=0.0,
            reference_windows_reliable=True,
        )
        broad_touch = tuple((row_index, column_index) for row_index in range(6) for column_index in range(6))
        records = [
            make_record(1, 102, movement_present=True, touched_grid_coordinates=broad_touch),
            make_record(2, 103, movement_present=True, touched_grid_coordinates=broad_touch),
            make_record(3, 104, movement_present=True, touched_grid_coordinates=broad_touch),
        ]
        sampled_frames = [
            make_stage3_art_state_sample(88, changed=False),
            make_stage3_art_state_sample(90, changed=False),
            make_stage3_art_state_sample(92, changed=False),
            make_stage3_art_state_sample(94, changed=False),
            make_stage3_art_state_sample(96, changed=False),
            make_stage3_art_state_sample(98, changed=False),
            make_stage3_art_state_sample(102, changed_cells=broad_touch),
            make_stage3_art_state_sample(103, changed_cells=broad_touch),
            make_stage3_art_state_sample(104, changed_cells=broad_touch),
        ]

        refined = refine_stage5_sub_slices(
            [screened_union],
            [valid_slice],
            records,
            sampled_frames=sampled_frames,
            settings=make_settings(),
            minimum_subdivision_frames=15,
        )

        self.assertEqual(len(refined), 1)
        self.assertEqual(refined[0].start_frame, 100)
        self.assertEqual(refined[0].end_frame, 120)
        self.assertIsNotNone(refined[0].stage5_debug)
        self.assertEqual(refined[0].stage5_debug['end_trim_reason'], 'minimum_clip_length_veto')

    def test_stage5_leaves_non_valid_band_handoffs_unchanged_for_future_boundary_work(self) -> None:
        unresolved_band = ClassifiedTimeSlice(
            slice_index=1,
            parent_union_index=1,
            slice_level=0,
            start_frame=300,
            end_frame=360,
            start_time='00:00:10:00',
            end_time='00:00:12:00',
            footprint=frozenset({(2, 2)}),
            footprint_size=1,
            within_slice_record_count=1,
            classification='undetermined',
            reason='probe_holding_band',
            lasting_change_evidence_score=0.40,
            before_reference_activity=0.10,
            after_reference_activity=0.10,
            reference_windows_reliable=True,
        )

        refined = refine_stage5_sub_slices([], [unresolved_band], [], sampled_frames=[], settings=make_settings(), minimum_subdivision_frames=15)

        self.assertEqual(len(refined), 1)
        self.assertEqual(refined[0], unresolved_band)

# SECTION H - Stage 6 Final Candidate Range Assembly Tests
# ============================================================


class FinalCandidateRangeAssemblyTests(unittest.TestCase):
    def test_stage6_keeps_valid_slices_and_retains_adjacent_minimum_size_boundary_edges(self) -> None:
        refined_slices = [
            ClassifiedTimeSlice(
                slice_index=1,
                parent_union_index=1,
                slice_level=1,
                start_frame=300,
                end_frame=315,
                start_time="00:00:10:00",
                end_time="00:00:10:15",
                footprint=frozenset({(2, 2)}),
                footprint_size=1,
                within_slice_record_count=1,
                classification="boundary",
                reason="minimum_subdivision_size_reached",
                lasting_change_evidence_score=0.66,
                before_reference_activity=0.72,
                after_reference_activity=0.68,
                reference_windows_reliable=True,
            ),
            ClassifiedTimeSlice(
                slice_index=2,
                parent_union_index=1,
                slice_level=1,
                start_frame=315,
                end_frame=330,
                start_time="00:00:10:15",
                end_time="00:00:11:00",
                footprint=frozenset({(2, 3), (3, 3)}),
                footprint_size=8,
                within_slice_record_count=2,
                classification="valid",
                reason="slice_activity_supported",
                lasting_change_evidence_score=0.45,
                before_reference_activity=0.03,
                after_reference_activity=0.02,
                reference_windows_reliable=True,
            ),
        ]

        final_ranges = assemble_stage6_candidate_ranges(refined_slices)

        self.assertEqual(len(final_ranges), 1)
        self.assertEqual(final_ranges[0].start_frame, 300)
        self.assertEqual(final_ranges[0].end_frame, 330)
        self.assertTrue(final_ranges[0].includes_boundary)
        self.assertEqual(final_ranges[0].boundary_count, 1)
        self.assertEqual(final_ranges[0].source_classifications, ("boundary", "valid"))

    def test_stage6_merges_valid_and_boundary_across_short_internal_gap(self) -> None:
        refined_slices = [
            ClassifiedTimeSlice(
                slice_index=1,
                parent_union_index=1,
                slice_level=1,
                start_frame=300,
                end_frame=315,
                start_time="00:00:10:00",
                end_time="00:00:10:15",
                footprint=frozenset({(2, 2), (2, 3), (3, 2)}),
                footprint_size=3,
                within_slice_record_count=1,
                classification="boundary",
                reason="minimum_subdivision_size_reached",
                lasting_change_evidence_score=0.69,
                before_reference_activity=0.72,
                after_reference_activity=0.68,
                reference_windows_reliable=True,
            ),
            ClassifiedTimeSlice(
                slice_index=2,
                parent_union_index=1,
                slice_level=1,
                start_frame=328,
                end_frame=343,
                start_time="00:00:10:28",
                end_time="00:00:11:13",
                footprint=frozenset({(2, 2), (2, 3), (3, 2), (3, 3)}),
                footprint_size=4,
                within_slice_record_count=1,
                classification="valid",
                reason="slice_activity_supported",
                lasting_change_evidence_score=0.74,
                before_reference_activity=0.05,
                after_reference_activity=0.03,
                reference_windows_reliable=True,
            ),
        ]

        final_ranges = assemble_stage6_candidate_ranges(refined_slices)

        self.assertEqual(len(final_ranges), 1)
        self.assertEqual(final_ranges[0].start_frame, 300)
        self.assertEqual(final_ranges[0].end_frame, 343)
        self.assertEqual(final_ranges[0].source_classifications, ("boundary", "valid"))

    def test_stage6_drops_isolated_boundary_minimum_size_slice(self) -> None:
        refined_slices = [
            ClassifiedTimeSlice(
                slice_index=1,
                parent_union_index=1,
                slice_level=1,
                start_frame=300,
                end_frame=315,
                start_time="00:00:10:00",
                end_time="00:00:10:15",
                footprint=frozenset({(2, 2)}),
                footprint_size=1,
                within_slice_record_count=1,
                classification="boundary",
                reason="minimum_subdivision_size_reached",
                lasting_change_evidence_score=0.66,
                before_reference_activity=0.72,
                after_reference_activity=0.68,
                reference_windows_reliable=True,
            )
        ]

        final_ranges = assemble_stage6_candidate_ranges(refined_slices)
        self.assertEqual(final_ranges, [])

    def test_stage6_drops_boundary_only_cluster_even_when_strong(self) -> None:
        refined_slices = [
            ClassifiedTimeSlice(
                slice_index=1,
                parent_union_index=3,
                slice_level=1,
                start_frame=500,
                end_frame=515,
                start_time="00:00:16:20",
                end_time="00:00:17:05",
                footprint=frozenset({(4, 4)}),
                footprint_size=8,
                within_slice_record_count=1,
                classification="boundary",
                reason="minimum_subdivision_size_reached",
                lasting_change_evidence_score=0.69,
                before_reference_activity=0.70,
                after_reference_activity=0.68,
                reference_windows_reliable=True,
            ),
            ClassifiedTimeSlice(
                slice_index=2,
                parent_union_index=3,
                slice_level=1,
                start_frame=515,
                end_frame=530,
                start_time="00:00:17:05",
                end_time="00:00:17:20",
                footprint=frozenset({(4, 5)}),
                footprint_size=8,
                within_slice_record_count=1,
                classification="boundary",
                reason="minimum_subdivision_size_reached",
                lasting_change_evidence_score=0.68,
                before_reference_activity=0.69,
                after_reference_activity=0.66,
                reference_windows_reliable=True,
            ),
            ClassifiedTimeSlice(
                slice_index=3,
                parent_union_index=3,
                slice_level=1,
                start_frame=530,
                end_frame=545,
                start_time="00:00:17:20",
                end_time="00:00:18:05",
                footprint=frozenset({(4, 6)}),
                footprint_size=8,
                within_slice_record_count=1,
                classification="boundary",
                reason="minimum_subdivision_size_reached",
                lasting_change_evidence_score=0.70,
                before_reference_activity=0.68,
                after_reference_activity=0.67,
                reference_windows_reliable=True,
            ),
            ClassifiedTimeSlice(
                slice_index=4,
                parent_union_index=3,
                slice_level=1,
                start_frame=545,
                end_frame=560,
                start_time="00:00:18:05",
                end_time="00:00:18:20",
                footprint=frozenset({(4, 7)}),
                footprint_size=8,
                within_slice_record_count=1,
                classification="boundary",
                reason="minimum_subdivision_size_reached",
                lasting_change_evidence_score=0.69,
                before_reference_activity=0.67,
                after_reference_activity=0.66,
                reference_windows_reliable=True,
            ),
        ]

        final_ranges = assemble_stage6_candidate_ranges(refined_slices)
        self.assertEqual(final_ranges, [])

    def test_stage6_drops_short_boundary_only_cluster_even_with_large_footprint(self) -> None:
        refined_slices = [
            ClassifiedTimeSlice(
                slice_index=1,
                parent_union_index=9,
                slice_level=1,
                start_frame=900,
                end_frame=908,
                start_time="00:00:30:00",
                end_time="00:00:30:08",
                footprint=frozenset((row_index, column_index) for row_index in range(12) for column_index in range(12)),
                footprint_size=TOTAL_GRID_BLOCKS,
                within_slice_record_count=1,
                classification="boundary",
                reason="minimum_subdivision_size_reached",
                lasting_change_evidence_score=0.655,
                before_reference_activity=0.72,
                after_reference_activity=0.70,
                reference_windows_reliable=True,
            ),
            ClassifiedTimeSlice(
                slice_index=2,
                parent_union_index=9,
                slice_level=1,
                start_frame=908,
                end_frame=916,
                start_time="00:00:30:08",
                end_time="00:00:30:16",
                footprint=frozenset({(4, 4), (4, 5)}),
                footprint_size=8,
                within_slice_record_count=1,
                classification="boundary",
                reason="minimum_subdivision_size_reached",
                lasting_change_evidence_score=0.665,
                before_reference_activity=0.72,
                after_reference_activity=0.70,
                reference_windows_reliable=True,
            ),
        ]

        final_ranges = assemble_stage6_candidate_ranges(refined_slices)
        self.assertEqual(final_ranges, [])

    def test_stage6_merges_low_overlap_boundary_slice_into_valid_boundary_when_gap_is_small(self) -> None:
        refined_slices = [
            ClassifiedTimeSlice(
                slice_index=1,
                parent_union_index=15,
                slice_level=1,
                start_frame=1000,
                end_frame=1012,
                start_time="00:00:33:10",
                end_time="00:00:33:22",
                footprint=frozenset({(7, 7), (7, 8), (8, 7)}),
                footprint_size=8,
                within_slice_record_count=1,
                classification="boundary",
                reason="minimum_subdivision_size_reached",
                lasting_change_evidence_score=0.62,
                before_reference_activity=0.66,
                after_reference_activity=0.55,
                reference_windows_reliable=True,
            ),
            ClassifiedTimeSlice(
                slice_index=2,
                parent_union_index=15,
                slice_level=1,
                start_frame=1024,
                end_frame=1036,
                start_time="00:00:34:04",
                end_time="00:00:34:16",
                footprint=frozenset({(1, 11), (2, 11), (3, 11), (4, 11)}),
                footprint_size=19,
                within_slice_record_count=1,
                classification="valid",
                reason="slice_activity_supported",
                lasting_change_evidence_score=0.76,
                before_reference_activity=0.58,
                after_reference_activity=0.21,
                reference_windows_reliable=True,
            ),
        ]

        final_ranges = assemble_stage6_candidate_ranges(refined_slices)

        self.assertEqual(len(final_ranges), 1)
        self.assertEqual(final_ranges[0].start_frame, 1000)
        self.assertEqual(final_ranges[0].end_frame, 1036)
        self.assertEqual(final_ranges[0].source_classifications, ("boundary", "valid"))

    def test_stage6_merges_contiguous_valid_ranges_across_union_boundaries(self) -> None:
        refined_slices = [
            ClassifiedTimeSlice(
                slice_index=1,
                parent_union_index=2,
                slice_level=1,
                start_frame=400,
                end_frame=415,
                start_time="00:00:13:10",
                end_time="00:00:13:25",
                footprint=frozenset({(4, 4)}),
                footprint_size=1,
                within_slice_record_count=1,
                classification="valid",
                reason="slice_activity_supported",
                lasting_change_evidence_score=0.42,
                before_reference_activity=0.02,
                after_reference_activity=0.02,
                reference_windows_reliable=True,
            ),
            ClassifiedTimeSlice(
                slice_index=2,
                parent_union_index=8,
                slice_level=1,
                start_frame=415,
                end_frame=430,
                start_time="00:00:13:25",
                end_time="00:00:14:10",
                footprint=frozenset({(4, 5)}),
                footprint_size=1,
                within_slice_record_count=1,
                classification="valid",
                reason="slice_activity_supported",
                lasting_change_evidence_score=0.41,
                before_reference_activity=0.02,
                after_reference_activity=0.02,
                reference_windows_reliable=True,
            ),
        ]

        final_ranges = assemble_stage6_candidate_ranges(refined_slices)

        self.assertEqual(len(final_ranges), 1)
        self.assertEqual(final_ranges[0].start_frame, 400)
        self.assertEqual(final_ranges[0].end_frame, 430)
        self.assertFalse(final_ranges[0].includes_boundary)
        self.assertEqual(final_ranges[0].source_classifications, ("valid",))


# ============================================================
# SECTION I - Staged Debug Artifact Output Tests
# ============================================================


class StagedDebugArtifactOutputTests(unittest.TestCase):
    def test_build_staged_debug_summary_lines_reports_stage_counts_and_ranges(self) -> None:
        debug_payload = {
            'movement_evidence_records': [{'record_index': 1}, {'record_index': 2}],
            'movement_spans': [
                {'span_index': 1, 'start_time': '00:00:01:00', 'end_time': '00:00:02:00'},
            ],
            'candidate_unions': [
                {'union_index': 1, 'start_time': '00:00:01:00', 'end_time': '00:00:02:15'},
                {'union_index': 2, 'start_time': '00:00:03:00', 'end_time': '00:00:03:20'},
            ],
            'screened_candidate_unions': [
                {'candidate_union_index': 1, 'surviving': True, 'provisional_survival': False},
                {'candidate_union_index': 2, 'surviving': False, 'provisional_survival': True},
            ],
            'classified_time_slices': [
                {'classification': 'valid'},
                {'classification': 'invalid'},
                {'classification': 'undetermined'},
            ],
            'refined_sub_slices': [
                {'classification': 'valid'},
                {'classification': 'boundary'},
                {'classification': 'invalid'},
            ],
            'final_candidate_ranges': [
                {
                    'range_index': 1,
                    'start_time': '00:00:10:00',
                    'end_time': '00:00:11:00',
                    'source_classifications': ['valid', 'boundary'],
                    'includes_boundary': True,
                    'boundary_count': 1,
                }
            ],
            'candidate_clips': [
                {
                    'clip_index': 1,
                    'clip_start': '00:00:09:28',
                    'clip_end': '00:00:11:04',
                    'activity_start': '00:00:10:00',
                    'activity_end': '00:00:11:00',
                }
            ],
            'stage_timings': [
                {
                    'stage_label': 'Stage 4/8 - Collecting Stage 3 art-state samples',
                    'elapsed_hhmmss': '07:12',
                    'item_count': 6568,
                },
                {
                    'stage_label': 'Total staged detector time',
                    'elapsed_hhmmss': '52:30',
                }
            ],
        }

        summary_lines = build_staged_debug_summary_lines(debug_payload)
        summary_text = '\n'.join(summary_lines)

        self.assertIn('Movement evidence', summary_text)
        self.assertIn('- Movement evidence records created: 2', summary_text)
        self.assertIn('Stage 1 - Movement spans', summary_text)
        self.assertIn('- Movement spans created: 1', summary_text)
        self.assertIn('- Span 1: 00:00:01:00 to 00:00:02:00', summary_text)
        self.assertIn('Stage 2 - Candidate unions', summary_text)
        self.assertIn('- Candidate unions created: 2', summary_text)
        self.assertIn('- Union 2: 00:00:03:00 to 00:00:03:20', summary_text)
        self.assertIn('Stage 3 - Union screening', summary_text)
        self.assertIn('- Survived: 1', summary_text)
        self.assertIn('- Rejected: 1', summary_text)
        self.assertIn('- Provisional survivals: 1', summary_text)
        self.assertIn('Stage 4 - Probe-bands', summary_text)
        self.assertIn('- Probe-bands created: 3', summary_text)
        self.assertIn('- Valid: 1', summary_text)
        self.assertIn('- Invalid: 1', summary_text)
        self.assertIn('- Undetermined: 1', summary_text)
        self.assertIn('Stage 5 - Local boundary refinement', summary_text)
        self.assertIn('- Refined slices produced: 3', summary_text)
        self.assertIn('- Boundary: 1', summary_text)
        self.assertIn('- Undetermined remaining after refinement: 0', summary_text)
        self.assertIn('Stage 6 - Final retained ranges', summary_text)
        self.assertIn('- Final retained ranges built from valid material: 1', summary_text)
        self.assertIn('- Ranges that include boundary support: 1', summary_text)
        self.assertIn('- Candidate clips produced: 1', summary_text)
        self.assertIn('Final retained ranges', summary_text)
        self.assertIn('  Built from: valid + boundary support', summary_text)
        self.assertIn('  Boundary slices attached: 1', summary_text)
        self.assertIn('Candidate clips', summary_text)
        self.assertIn('  Activity inside clip: 00:00:10:00 to 00:00:11:00', summary_text)

    def test_write_staged_debug_artifacts_uses_stage_labeled_filenames(self) -> None:
        debug_payload = {
            'movement_evidence_records': [],
            'movement_spans': [],
            'candidate_unions': [],
            'screened_candidate_unions': [],
            'stage3_screening_traces': [],
            'classified_time_slices': [],
            'stage4_subregion_debug': [],
            'refined_sub_slices': [],
            'final_candidate_ranges': [],
            'candidate_clips': [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            debug_stem = Path(tmpdir) / 'Backpack'
            stage3_samples = [make_stage3_art_state_sample(12, changed=True), make_stage3_art_state_sample(14)]
            cache_path = write_reusable_stage3_art_state_sample_cache(debug_stem, stage3_samples)
            output_paths = write_staged_debug_artifacts(debug_stem, debug_payload)

            self.assertEqual(output_paths['movement_evidence_records'].name, 'Backpack - Stage 1A - Movement Evidence Record.json')
            self.assertEqual(output_paths['movement_spans'].name, 'Backpack - Stage 1B - Movement Spans.json')
            self.assertEqual(output_paths['reusable_stage3_art_state_samples'].name, 'Backpack - Stage 1C - Reusable Stage 2B Frame Payload.npz')
            self.assertEqual(output_paths['candidate_unions'].name, 'Backpack - Stage 2A - Candidate Union Record.json')
            self.assertEqual(output_paths['screened_candidate_unions'].name, 'Backpack - Stage 3A - Union Screening.json')
            self.assertEqual(output_paths['stage3_screening_traces'].name, 'Backpack - Stage 3B - Screening Trace [Step 1 + Snapshot Rescue].json')
            self.assertEqual(output_paths['classified_time_slices'].name, 'Backpack - Stage 4 - Probe-Band Classifications.json')
            self.assertEqual(output_paths['stage4_subregion_debug'].name, 'Backpack - Stage 4B - Probe Debug Output.json')
            self.assertEqual(output_paths['refined_sub_slices'].name, 'Backpack - Stage 5 - Local Boundary Refinement Output.json')
            self.assertEqual(output_paths['final_candidate_ranges'].name, 'Backpack - Stage 6A - Candidate Ranges [Pre-Filters].json')
            self.assertEqual(output_paths['candidate_clips'].name, 'Backpack - Stage 6B - Candidate Pre-Clips [Post-Filters].json')
            self.assertEqual(output_paths['summary'].name, 'Backpack - Debug Summary.txt')
            self.assertTrue(cache_path.exists())

            summary_path = output_paths['summary']
            self.assertTrue(summary_path.exists())
            summary_text = summary_path.read_text(encoding='utf-8')
            self.assertIn('Staged detector summary', summary_text)
            self.assertIn('Stage 6 - Final retained ranges', summary_text)
            self.assertIn('- Final retained ranges built from valid material: 0', summary_text)

    def test_reusable_stage3_art_state_sample_cache_round_trips_from_movement_evidence_path(self) -> None:
        stage3_samples = [make_stage3_art_state_sample(12, changed=True), make_stage3_art_state_sample(14)]

        with tempfile.TemporaryDirectory() as tmpdir:
            debug_stem = Path(tmpdir) / 'Backpack'
            movement_evidence_path = debug_stem.with_name('Backpack - Stage 1A - Movement Evidence Record.json')
            movement_evidence_path.write_text('[]', encoding='utf-8')
            write_reusable_stage3_art_state_sample_cache(debug_stem, stage3_samples)

            loaded_from_cache = load_reusable_stage3_art_state_sample_cache(
                debug_stem.with_name('Backpack - Stage 1C - Reusable Stage 2B Frame Payload.npz')
            )
            loaded_from_movement_evidence_path = load_precomputed_stage3_art_state_sample_cache_from_movement_evidence_path(
                movement_evidence_path
            )

            self.assertEqual(len(loaded_from_cache), 2)
            self.assertEqual([sample['frame_index'] for sample in loaded_from_cache], [12, 14])
            self.assertIsNotNone(loaded_from_movement_evidence_path)
            self.assertEqual([sample['frame_index'] for sample in loaded_from_movement_evidence_path], [12, 14])
            self.assertEqual(loaded_from_cache[0]['canvas_shape'], stage3_samples[0]['canvas_shape'])
            self.assertTrue(np.array_equal(loaded_from_cache[0]['art_gray'], stage3_samples[0]['art_gray']))

    def test_reusable_stage3_art_state_sample_cache_rejects_scan_resolution_mismatch(self) -> None:
        stage3_samples = [make_stage3_art_state_sample(12, changed=True), make_stage3_art_state_sample(14)]
        expected_metadata = {
            'mode': 'half',
            'analysis_canvas_height': 120,
            'analysis_canvas_width': 120,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            debug_stem = Path(tmpdir) / 'Backpack'
            write_reusable_stage3_art_state_sample_cache(
                debug_stem,
                stage3_samples,
                scan_resolution_metadata={
                    'mode': 'full',
                    'analysis_canvas_height': 120,
                    'analysis_canvas_width': 120,
                },
            )

            with self.assertRaisesRegex(RuntimeError, "scan resolution 'full'.*uses 'half'"):
                load_reusable_stage3_art_state_sample_cache(
                    debug_stem.with_name('Backpack - Stage 1C - Reusable Stage 2B Frame Payload.npz'),
                    expected_scan_resolution_metadata=expected_metadata,
                )


if __name__ == "__main__":
    unittest.main()




































