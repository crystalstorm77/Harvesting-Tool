from __future__ import annotations

# ============================================================
# SECTION A - Imports And Helpers
# ============================================================

import unittest
from collections import deque

import cv2
import numpy as np

from harvesting_tool.detection import DetectorSettings, Timecode
from harvesting_tool.staged_detection import (
    CandidateUnion,
    ClassifiedTimeSlice,
    MovementEvidenceRecord,
    MovementSpan,
    ScreenedCandidateUnion,
    assemble_stage6_candidate_ranges,
    build_movement_evidence_record,
    build_stage1_movement_spans,
    build_stage2_candidate_unions,
    classify_stage4_time_slices,
    classify_stage5_minimum_size_leaf,
    refine_stage5_sub_slices,
    screen_stage3_candidate_unions,
)


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
    surviving: bool = True,
    provisional_survival: bool = False,
    screening_result: str = "surviving",
) -> ScreenedCandidateUnion:
    candidate_union = CandidateUnion(
        union_index=union_index,
        start_frame=start_frame,
        end_frame=end_frame,
        start_time=Timecode(total_frames=start_frame).to_hhmmssff(),
        end_time=Timecode(total_frames=end_frame).to_hhmmssff(),
        member_movement_spans=(),
        union_footprint=frozenset({(2, 2), (2, 3), (3, 2), (3, 3)}) if union_footprint_size else frozenset(),
        union_footprint_size=union_footprint_size,
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
    def test_stage1_movement_spans_backtrack_and_union_footprints(self) -> None:
        settings = make_settings()
        records = [
            make_record(1, 100, movement_present=True, weak_signal=True, touched_grid_coordinates=((0, 0),)),
            make_record(2, 102, movement_present=True, weak_signal=True, touched_grid_coordinates=((0, 1),)),
            make_record(3, 104, movement_present=True, opening_signal=True, touched_grid_coordinates=((1, 1),)),
            make_record(4, 106, movement_present=True, continuation_signal=True, touched_grid_coordinates=((1, 2),)),
            make_record(5, 108, movement_present=False),
            make_record(6, 110, movement_present=False),
        ]

        spans = build_stage1_movement_spans(records, settings)

        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertEqual(span.start_frame, 100)
        self.assertEqual(span.record_indices, (1, 2, 3, 4))
        self.assertEqual(span.footprint_size, 4)
        self.assertEqual(
            span.footprint,
            frozenset({(0, 0), (0, 1), (1, 1), (1, 2)}),
        )

    def test_stage1_movement_spans_discard_short_runs_below_minimum_length(self) -> None:
        settings = make_settings()
        records = [
            make_record(1, 200, movement_present=True, opening_signal=True, touched_grid_coordinates=((3, 3),)),
            make_record(2, 202, movement_present=False),
            make_record(3, 204, movement_present=False),
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

    def test_stage2_candidate_unions_split_on_short_temporal_gap_without_spatial_support(self) -> None:
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

    def test_stage2_candidate_unions_split_on_large_footprint_expansion_jump(self) -> None:
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

        self.assertEqual(len(unions), 2)
        self.assertEqual(unions[0].member_movement_spans, (spans[0], spans[1]))
        self.assertEqual(unions[1].member_movement_spans, (spans[2],))
# ============================================================
# SECTION E - Stage 3 Candidate Union Screening Tests
# ============================================================


class CandidateUnionScreeningTests(unittest.TestCase):
    def test_stage3_candidate_union_survives_with_strong_union_activity_and_quiet_references(self) -> None:
        candidate_union = CandidateUnion(
            union_index=1,
            start_frame=300,
            end_frame=330,
            start_time="00:00:10:00",
            end_time="00:00:11:00",
            member_movement_spans=(),
            union_footprint=frozenset({(2, 2), (2, 3), (3, 2), (3, 3)}),
            union_footprint_size=4,
        )
        records = [
            make_record(1, 292, movement_present=False, movement_strength_score=0.02, temporal_persistence_score=0.02),
            make_record(2, 298, movement_present=False, movement_strength_score=0.03, temporal_persistence_score=0.02),
            make_record(3, 304, movement_present=True, movement_strength_score=0.62, temporal_persistence_score=0.58, spatial_extent_score=0.5),
            make_record(4, 316, movement_present=True, movement_strength_score=0.66, temporal_persistence_score=0.6, spatial_extent_score=0.5),
            make_record(5, 332, movement_present=False, movement_strength_score=0.01, temporal_persistence_score=0.01),
            make_record(6, 338, movement_present=False, movement_strength_score=0.02, temporal_persistence_score=0.01),
        ]

        screened = screen_stage3_candidate_unions([candidate_union], records)

        self.assertEqual(len(screened), 1)
        self.assertEqual(screened[0].screening_result, "surviving")
        self.assertTrue(screened[0].surviving)
        self.assertFalse(screened[0].provisional_survival)

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

    def test_stage3_candidate_union_survives_provisionally_when_reference_windows_are_unreliable(self) -> None:
        candidate_union = CandidateUnion(
            union_index=1,
            start_frame=300,
            end_frame=330,
            start_time="00:00:10:00",
            end_time="00:00:11:00",
            member_movement_spans=(),
            union_footprint=frozenset({(2, 2), (2, 3), (3, 2), (3, 3)}),
            union_footprint_size=4,
        )
        records = [
            make_record(1, 304, movement_present=True, movement_strength_score=0.65, temporal_persistence_score=0.6, spatial_extent_score=0.55),
            make_record(2, 316, movement_present=True, movement_strength_score=0.68, temporal_persistence_score=0.62, spatial_extent_score=0.55),
            make_record(3, 336, movement_present=False, movement_strength_score=0.02, temporal_persistence_score=0.01),
        ]

        screened = screen_stage3_candidate_unions([candidate_union], records)

        self.assertEqual(screened[0].screening_result, "provisional_surviving")
        self.assertTrue(screened[0].provisional_survival)
        self.assertTrue(screened[0].surviving)

    def test_stage3_candidate_union_rejects_when_reference_windows_are_too_active(self) -> None:
        candidate_union = CandidateUnion(
            union_index=1,
            start_frame=300,
            end_frame=330,
            start_time="00:00:10:00",
            end_time="00:00:11:00",
            member_movement_spans=(),
            union_footprint=frozenset({(2, 2), (2, 3), (3, 2), (3, 3)}),
            union_footprint_size=4,
        )
        records = [
            make_record(1, 292, movement_present=True, movement_strength_score=0.42, temporal_persistence_score=0.4),
            make_record(2, 298, movement_present=True, movement_strength_score=0.4, temporal_persistence_score=0.38),
            make_record(3, 304, movement_present=True, movement_strength_score=0.38, temporal_persistence_score=0.36, spatial_extent_score=0.35),
            make_record(4, 316, movement_present=True, movement_strength_score=0.37, temporal_persistence_score=0.35, spatial_extent_score=0.35),
            make_record(5, 332, movement_present=True, movement_strength_score=0.42, temporal_persistence_score=0.40),
            make_record(6, 338, movement_present=True, movement_strength_score=0.43, temporal_persistence_score=0.41),
        ]

        screened = screen_stage3_candidate_unions([candidate_union], records)

        self.assertEqual(screened[0].screening_result, "rejected")
        self.assertEqual(screened[0].reason, "reference_windows_too_active")


# ============================================================
# SECTION F - Stage 4 Time Slice Classification Tests
# ============================================================


class TimeSliceClassificationTests(unittest.TestCase):
    def test_stage4_classifies_valid_and_invalid_halves_of_surviving_union(self) -> None:
        screened_union = make_screened_union()
        records = [
            make_record(1, 304, movement_present=True, movement_strength_score=0.84, temporal_persistence_score=0.82, spatial_extent_score=0.90, touched_grid_coordinates=((2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3))),
            make_record(2, 318, movement_present=True, movement_strength_score=0.82, temporal_persistence_score=0.80, spatial_extent_score=0.88, touched_grid_coordinates=((4, 3), (4, 4), (4, 5), (5, 3), (5, 4), (5, 5), (6, 3), (6, 4))),
            make_record(3, 332, movement_present=False, movement_strength_score=0.04, temporal_persistence_score=0.03, spatial_extent_score=0.02),
            make_record(4, 338, movement_present=False, movement_strength_score=0.03, temporal_persistence_score=0.02, spatial_extent_score=0.02),
            make_record(5, 292, movement_present=False, movement_strength_score=0.02, temporal_persistence_score=0.02),
            make_record(6, 298, movement_present=False, movement_strength_score=0.02, temporal_persistence_score=0.02),
            make_record(7, 366, movement_present=False, movement_strength_score=0.01, temporal_persistence_score=0.01),
            make_record(8, 378, movement_present=False, movement_strength_score=0.01, temporal_persistence_score=0.01),
        ]

        slices = classify_stage4_time_slices([screened_union], records)

        self.assertEqual(len(slices), 2)
        self.assertEqual([slice_info.classification for slice_info in slices], ["valid", "invalid"])
        self.assertEqual(slices[0].footprint_size, 15)
        self.assertEqual(slices[1].footprint_size, 0)

    def test_stage4_marks_mixed_slice_as_undetermined(self) -> None:
        screened_union = make_screened_union(start_frame=300, end_frame=330)
        records = [
            make_record(1, 304, movement_present=True, movement_strength_score=0.24, temporal_persistence_score=0.22, spatial_extent_score=0.2, touched_grid_coordinates=((2, 2),)),
            make_record(2, 312, movement_present=True, movement_strength_score=0.17, temporal_persistence_score=0.15, spatial_extent_score=0.14, touched_grid_coordinates=((2, 3),)),
            make_record(3, 292, movement_present=False, movement_strength_score=0.12, temporal_persistence_score=0.1),
            make_record(4, 298, movement_present=False, movement_strength_score=0.11, temporal_persistence_score=0.1),
            make_record(5, 320, movement_present=False, movement_strength_score=0.12, temporal_persistence_score=0.1),
            make_record(6, 324, movement_present=False, movement_strength_score=0.11, temporal_persistence_score=0.1),
        ]

        slices = classify_stage4_time_slices([screened_union], records)

        self.assertEqual(len(slices), 2)
        self.assertEqual(slices[0].classification, "undetermined")
        self.assertEqual(slices[0].reason, "mixed_slice_evidence")

    def test_stage4_marks_boundary_slice_with_one_quiet_reference_as_undetermined(self) -> None:
        screened_union = make_screened_union(start_frame=300, end_frame=420)
        records = [
            make_record(1, 292, movement_present=False, movement_strength_score=0.02, temporal_persistence_score=0.02),
            make_record(2, 298, movement_present=False, movement_strength_score=0.03, temporal_persistence_score=0.02),
            make_record(3, 334, movement_present=True, movement_strength_score=0.80, temporal_persistence_score=0.78, spatial_extent_score=0.88, touched_grid_coordinates=((2, 2), (2, 3), (3, 2), (3, 3), (4, 2), (4, 3))),
            make_record(4, 346, movement_present=True, movement_strength_score=0.82, temporal_persistence_score=0.80, spatial_extent_score=0.90, touched_grid_coordinates=((4, 2), (4, 3), (5, 2), (5, 3), (6, 2), (6, 3))),
            make_record(5, 364, movement_present=True, movement_strength_score=0.76, temporal_persistence_score=0.74, spatial_extent_score=0.86, touched_grid_coordinates=((3, 3), (3, 4), (4, 3), (4, 4))),
            make_record(6, 368, movement_present=True, movement_strength_score=0.78, temporal_persistence_score=0.76, spatial_extent_score=0.88, touched_grid_coordinates=((4, 3), (4, 4), (5, 3), (5, 4))),
            make_record(7, 392, movement_present=True, movement_strength_score=0.73, temporal_persistence_score=0.69, spatial_extent_score=0.64),
            make_record(8, 404, movement_present=True, movement_strength_score=0.71, temporal_persistence_score=0.67, spatial_extent_score=0.63),
        ]

        slices = classify_stage4_time_slices([screened_union], records)

        self.assertEqual(len(slices), 2)
        self.assertEqual(slices[0].classification, "undetermined")
        self.assertEqual(slices[0].reason, "mixed_reference_activity")
        self.assertEqual(slices[1].classification, "undetermined")

    def test_stage4_accepts_boundary_slice_that_just_misses_full_contrast_floor(self) -> None:
        screened_union = make_screened_union(start_frame=300, end_frame=420)
        records = [
            make_record(1, 304, movement_present=False, movement_strength_score=0.03, temporal_persistence_score=0.03),
            make_record(2, 312, movement_present=False, movement_strength_score=0.03, temporal_persistence_score=0.03),
            make_record(3, 352, movement_present=True, movement_strength_score=0.76, temporal_persistence_score=0.74, spatial_extent_score=0.70),
            make_record(4, 358, movement_present=True, movement_strength_score=0.76, temporal_persistence_score=0.74, spatial_extent_score=0.70),
            make_record(5, 364, movement_present=True, movement_strength_score=0.78, temporal_persistence_score=0.76, spatial_extent_score=0.80, touched_grid_coordinates=((2, 2), (2, 3), (3, 2), (3, 3))),
            make_record(6, 376, movement_present=True, movement_strength_score=0.80, temporal_persistence_score=0.78, spatial_extent_score=0.82, touched_grid_coordinates=((3, 2), (3, 3), (4, 2), (4, 3))),
            make_record(7, 392, movement_present=True, movement_strength_score=0.76, temporal_persistence_score=0.74, spatial_extent_score=0.72, touched_grid_coordinates=((4, 2), (4, 3), (5, 2), (5, 3))),
            make_record(8, 404, movement_present=True, movement_strength_score=0.78, temporal_persistence_score=0.76, spatial_extent_score=0.74, touched_grid_coordinates=((5, 2), (5, 3), (6, 2), (6, 3))),
            make_record(9, 424, movement_present=False, movement_strength_score=0.01, temporal_persistence_score=0.01),
            make_record(10, 428, movement_present=False, movement_strength_score=0.01, temporal_persistence_score=0.01),
        ]

        slices = classify_stage4_time_slices([screened_union], records)

        self.assertEqual(len(slices), 2)
        self.assertEqual(slices[1].classification, "valid")
        self.assertEqual(slices[1].reason, "slice_activity_supported")

    def test_stage4_marks_strong_slice_with_active_references_as_undetermined(self) -> None:
        screened_union = make_screened_union(start_frame=300, end_frame=420)
        records = [
            make_record(1, 304, movement_present=True, movement_strength_score=0.78, temporal_persistence_score=0.76, spatial_extent_score=0.86, touched_grid_coordinates=((2, 2), (2, 3), (3, 2), (3, 3))),
            make_record(2, 332, movement_present=True, movement_strength_score=0.80, temporal_persistence_score=0.78, spatial_extent_score=0.88, touched_grid_coordinates=((3, 2), (3, 3), (4, 2), (4, 3))),
            make_record(3, 292, movement_present=True, movement_strength_score=0.68, temporal_persistence_score=0.66, spatial_extent_score=0.60),
            make_record(4, 298, movement_present=True, movement_strength_score=0.69, temporal_persistence_score=0.67, spatial_extent_score=0.61),
            make_record(5, 364, movement_present=True, movement_strength_score=0.67, temporal_persistence_score=0.65, spatial_extent_score=0.60),
            make_record(6, 368, movement_present=True, movement_strength_score=0.68, temporal_persistence_score=0.66, spatial_extent_score=0.61),
        ]

        slices = classify_stage4_time_slices([screened_union], records)

        self.assertEqual(len(slices), 2)
        self.assertEqual(slices[0].classification, "undetermined")
        self.assertEqual(slices[0].reason, "reference_windows_too_active")
    def test_stage4_ignores_rejected_unions(self) -> None:
        rejected_union = make_screened_union(surviving=False, screening_result="rejected")
        records = [
            make_record(1, 304, movement_present=True, movement_strength_score=0.7, temporal_persistence_score=0.65),
            make_record(2, 316, movement_present=True, movement_strength_score=0.68, temporal_persistence_score=0.62),
        ]

        slices = classify_stage4_time_slices([rejected_union], records)
        self.assertEqual(slices, [])


# ============================================================
# SECTION G - Stage 5 Recursive Sub-Slice Refinement Tests
# ============================================================


class SubSliceRefinementTests(unittest.TestCase):
    def test_stage5_refines_undetermined_slice_into_leaf_sub_slices(self) -> None:
        screened_union = make_screened_union(start_frame=300, end_frame=330)
        undetermined_slice = ClassifiedTimeSlice(
            slice_index=1,
            parent_union_index=screened_union.candidate_union.union_index,
            slice_level=0,
            start_frame=300,
            end_frame=330,
            start_time="00:00:10:00",
            end_time="00:00:11:00",
            footprint=frozenset({(2, 2), (2, 3)}),
            footprint_size=2,
            within_slice_record_count=2,
            classification="undetermined",
            reason="mixed_slice_evidence",
            lasting_change_evidence_score=0.2,
            before_reference_activity=0.11,
            after_reference_activity=0.11,
            reference_windows_reliable=True,
        )
        records = [
            make_record(1, 288, movement_present=False, movement_strength_score=0.02, temporal_persistence_score=0.02),
            make_record(2, 296, movement_present=False, movement_strength_score=0.02, temporal_persistence_score=0.02),
            make_record(3, 304, movement_present=True, movement_strength_score=0.86, temporal_persistence_score=0.84, spatial_extent_score=0.95, touched_grid_coordinates=((2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3))),
            make_record(4, 318, movement_present=False, movement_strength_score=0.01, temporal_persistence_score=0.01),
            make_record(5, 326, movement_present=False, movement_strength_score=0.01, temporal_persistence_score=0.01),
            make_record(6, 336, movement_present=False, movement_strength_score=0.01, temporal_persistence_score=0.01),
        ]

        refined = refine_stage5_sub_slices([screened_union], [undetermined_slice], records, minimum_subdivision_frames=15)

        self.assertEqual(len(refined), 2)
        self.assertEqual([slice_info.classification for slice_info in refined], ["valid", "invalid"])
        self.assertTrue(all(slice_info.slice_level == 1 for slice_info in refined))
        self.assertTrue(all(slice_info.parent_range == (300, 330) for slice_info in refined))

    def test_stage5_uses_local_reference_windows_for_small_sub_slices(self) -> None:
        screened_union = make_screened_union(start_frame=300, end_frame=330)
        undetermined_slice = ClassifiedTimeSlice(
            slice_index=1,
            parent_union_index=screened_union.candidate_union.union_index,
            slice_level=0,
            start_frame=300,
            end_frame=330,
            start_time="00:00:10:00",
            end_time="00:00:11:00",
            footprint=frozenset({(2, 2), (2, 3)}),
            footprint_size=2,
            within_slice_record_count=2,
            classification="undetermined",
            reason="mixed_reference_activity",
            lasting_change_evidence_score=0.3,
            before_reference_activity=0.02,
            after_reference_activity=0.6,
            reference_windows_reliable=True,
        )
        records = [
            make_record(1, 288, movement_present=False, movement_strength_score=0.02, temporal_persistence_score=0.02),
            make_record(2, 296, movement_present=False, movement_strength_score=0.03, temporal_persistence_score=0.02),
            make_record(3, 304, movement_present=True, movement_strength_score=0.88, temporal_persistence_score=0.86, spatial_extent_score=0.96, touched_grid_coordinates=((2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3))),
            make_record(4, 312, movement_present=True, movement_strength_score=0.86, temporal_persistence_score=0.84, spatial_extent_score=0.95, touched_grid_coordinates=((4, 3), (4, 4), (4, 5), (5, 3), (5, 4), (5, 5), (6, 3), (6, 4))),
            make_record(5, 318, movement_present=False, movement_strength_score=0.03, temporal_persistence_score=0.02),
            make_record(6, 326, movement_present=False, movement_strength_score=0.03, temporal_persistence_score=0.02),
            make_record(7, 332, movement_present=True, movement_strength_score=0.72, temporal_persistence_score=0.68, spatial_extent_score=0.62),
            make_record(8, 340, movement_present=True, movement_strength_score=0.73, temporal_persistence_score=0.69, spatial_extent_score=0.63),
        ]

        refined = refine_stage5_sub_slices([screened_union], [undetermined_slice], records, minimum_subdivision_frames=15)

        self.assertEqual(len(refined), 2)
        self.assertEqual(refined[0].classification, "valid")
        self.assertEqual(refined[0].reason, "slice_activity_supported")
    def test_stage5_refines_strong_active_reference_slice_instead_of_invalidating_it(self) -> None:
        screened_union = make_screened_union(start_frame=300, end_frame=420)
        undetermined_slice = ClassifiedTimeSlice(
            slice_index=1,
            parent_union_index=screened_union.candidate_union.union_index,
            slice_level=0,
            start_frame=300,
            end_frame=360,
            start_time="00:00:10:00",
            end_time="00:00:12:00",
            footprint=frozenset({(2, 2), (2, 3), (3, 2), (3, 3)}),
            footprint_size=4,
            within_slice_record_count=2,
            classification="undetermined",
            reason="reference_windows_too_active",
            lasting_change_evidence_score=0.72,
            before_reference_activity=0.72,
            after_reference_activity=0.70,
            reference_windows_reliable=True,
        )
        records = [
            make_record(1, 304, movement_present=True, movement_strength_score=0.84, temporal_persistence_score=0.82, spatial_extent_score=0.92, touched_grid_coordinates=((2, 2), (2, 3), (3, 2), (3, 3), (4, 2), (4, 3))),
            make_record(2, 318, movement_present=True, movement_strength_score=0.86, temporal_persistence_score=0.84, spatial_extent_score=0.94, touched_grid_coordinates=((3, 2), (3, 3), (4, 2), (4, 3), (5, 2), (5, 3))),
            make_record(3, 334, movement_present=True, movement_strength_score=0.74, temporal_persistence_score=0.72, spatial_extent_score=0.68),
            make_record(4, 346, movement_present=True, movement_strength_score=0.73, temporal_persistence_score=0.71, spatial_extent_score=0.67),
            make_record(5, 288, movement_present=True, movement_strength_score=0.70, temporal_persistence_score=0.68, spatial_extent_score=0.62),
            make_record(6, 296, movement_present=True, movement_strength_score=0.71, temporal_persistence_score=0.69, spatial_extent_score=0.63),
            make_record(7, 364, movement_present=False, movement_strength_score=0.02, temporal_persistence_score=0.02),
            make_record(8, 372, movement_present=False, movement_strength_score=0.02, temporal_persistence_score=0.02),
        ]

        refined = refine_stage5_sub_slices([screened_union], [undetermined_slice], records, minimum_subdivision_frames=15)

        self.assertTrue(any(slice_info.classification == "rocky" and slice_info.reason == "minimum_subdivision_size_reached" for slice_info in refined))
    def test_stage5_rescues_active_reference_minimum_size_leaf_when_union_has_valid_support(self) -> None:
        rescued = classify_stage5_minimum_size_leaf(
            ClassifiedTimeSlice(
                slice_index=1,
                parent_union_index=1,
                slice_level=3,
                start_frame=300,
                end_frame=315,
                start_time="00:00:10:00",
                end_time="00:00:10:15",
                footprint=frozenset({(2, 2), (2, 3)}),
                footprint_size=2,
                within_slice_record_count=2,
                classification="invalid",
                reason="reference_windows_too_active",
                lasting_change_evidence_score=0.58,
                before_reference_activity=0.76,
                after_reference_activity=0.75,
                reference_windows_reliable=True,
            ),
            allow_active_reference_rescue=True,
        )

        self.assertEqual(rescued.classification, "rocky")
        self.assertEqual(rescued.reason, "active_reference_minimum_size_reached")

    def test_stage5_stops_at_minimum_subdivision_size(self) -> None:
        screened_union = make_screened_union(start_frame=300, end_frame=312)
        undetermined_slice = ClassifiedTimeSlice(
            slice_index=1,
            parent_union_index=screened_union.candidate_union.union_index,
            slice_level=0,
            start_frame=300,
            end_frame=312,
            start_time="00:00:10:00",
            end_time="00:00:10:12",
            footprint=frozenset({(2, 2)}),
            footprint_size=1,
            within_slice_record_count=1,
            classification="undetermined",
            reason="mixed_slice_evidence",
            lasting_change_evidence_score=0.19,
            before_reference_activity=0.1,
            after_reference_activity=0.1,
            reference_windows_reliable=True,
        )

        refined = refine_stage5_sub_slices([screened_union], [undetermined_slice], [], minimum_subdivision_frames=15)

        self.assertEqual(len(refined), 1)
        self.assertEqual(refined[0].classification, "undetermined")
        self.assertEqual(refined[0].reason, "minimum_subdivision_size_reached")


# ============================================================
# SECTION H - Stage 6 Final Candidate Range Assembly Tests
# ============================================================


class FinalCandidateRangeAssemblyTests(unittest.TestCase):
    def test_stage6_keeps_valid_slices_and_retains_adjacent_minimum_size_rocky_edges(self) -> None:
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
                classification="rocky",
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
                footprint_size=2,
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
        self.assertTrue(final_ranges[0].includes_retained_undetermined)
        self.assertEqual(final_ranges[0].retained_undetermined_count, 1)
        self.assertEqual(final_ranges[0].source_classifications, ("rocky", "valid"))

    def test_stage6_drops_isolated_rocky_minimum_size_slice(self) -> None:
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
                classification="rocky",
                reason="minimum_subdivision_size_reached",
                lasting_change_evidence_score=0.66,
                before_reference_activity=0.72,
                after_reference_activity=0.68,
                reference_windows_reliable=True,
            )
        ]

        final_ranges = assemble_stage6_candidate_ranges(refined_slices)
        self.assertEqual(final_ranges, [])

    def test_stage6_keeps_rocky_only_cluster_when_cluster_strength_is_high(self) -> None:
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
                footprint_size=1,
                within_slice_record_count=1,
                classification="rocky",
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
                footprint_size=1,
                within_slice_record_count=1,
                classification="rocky",
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
                footprint_size=1,
                within_slice_record_count=1,
                classification="rocky",
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
                footprint_size=1,
                within_slice_record_count=1,
                classification="rocky",
                reason="minimum_subdivision_size_reached",
                lasting_change_evidence_score=0.69,
                before_reference_activity=0.67,
                after_reference_activity=0.66,
                reference_windows_reliable=True,
            ),
        ]

        final_ranges = assemble_stage6_candidate_ranges(refined_slices)

        self.assertEqual(len(final_ranges), 1)
        self.assertEqual(final_ranges[0].start_frame, 500)
        self.assertEqual(final_ranges[0].end_frame, 560)
        self.assertEqual(final_ranges[0].source_classifications, ("rocky",))

    def test_stage6_keeps_active_reference_rocky_reason_as_candidate(self) -> None:
        refined_slices = [
            ClassifiedTimeSlice(
                slice_index=1,
                parent_union_index=2,
                slice_level=1,
                start_frame=400,
                end_frame=415,
                start_time="00:00:13:10",
                end_time="00:00:13:25",
                footprint=frozenset({(2, 2)}),
                footprint_size=1,
                within_slice_record_count=1,
                classification="rocky",
                reason="active_reference_minimum_size_reached",
                lasting_change_evidence_score=0.62,
                before_reference_activity=0.75,
                after_reference_activity=0.74,
                reference_windows_reliable=True,
            ),
            ClassifiedTimeSlice(
                slice_index=2,
                parent_union_index=2,
                slice_level=1,
                start_frame=415,
                end_frame=430,
                start_time="00:00:13:25",
                end_time="00:00:14:10",
                footprint=frozenset({(2, 3)}),
                footprint_size=1,
                within_slice_record_count=1,
                classification="valid",
                reason="slice_activity_supported",
                lasting_change_evidence_score=0.80,
                before_reference_activity=0.74,
                after_reference_activity=0.02,
                reference_windows_reliable=True,
            ),
        ]

        final_ranges = assemble_stage6_candidate_ranges(refined_slices)

        self.assertEqual(len(final_ranges), 1)
        self.assertEqual(final_ranges[0].start_frame, 400)
        self.assertEqual(final_ranges[0].source_classifications, ("rocky", "valid"))

    def test_stage6_keeps_long_active_reference_rocky_cluster(self) -> None:
        refined_slices = [
            ClassifiedTimeSlice(
                slice_index=1,
                parent_union_index=4,
                slice_level=1,
                start_frame=800,
                end_frame=815,
                start_time="00:00:26:20",
                end_time="00:00:27:05",
                footprint=frozenset({(2, 2)}),
                footprint_size=1,
                within_slice_record_count=1,
                classification="rocky",
                reason="active_reference_minimum_size_reached",
                lasting_change_evidence_score=0.62,
                before_reference_activity=0.76,
                after_reference_activity=0.75,
                reference_windows_reliable=True,
            ),
            ClassifiedTimeSlice(
                slice_index=2,
                parent_union_index=4,
                slice_level=1,
                start_frame=815,
                end_frame=830,
                start_time="00:00:27:05",
                end_time="00:00:27:20",
                footprint=frozenset({(2, 3)}),
                footprint_size=1,
                within_slice_record_count=1,
                classification="rocky",
                reason="minimum_subdivision_size_reached",
                lasting_change_evidence_score=0.61,
                before_reference_activity=0.76,
                after_reference_activity=0.75,
                reference_windows_reliable=True,
            ),
            ClassifiedTimeSlice(
                slice_index=3,
                parent_union_index=4,
                slice_level=1,
                start_frame=830,
                end_frame=845,
                start_time="00:00:27:20",
                end_time="00:00:28:05",
                footprint=frozenset({(2, 4)}),
                footprint_size=1,
                within_slice_record_count=1,
                classification="rocky",
                reason="minimum_subdivision_size_reached",
                lasting_change_evidence_score=0.63,
                before_reference_activity=0.76,
                after_reference_activity=0.75,
                reference_windows_reliable=True,
            ),
            ClassifiedTimeSlice(
                slice_index=4,
                parent_union_index=4,
                slice_level=1,
                start_frame=845,
                end_frame=860,
                start_time="00:00:28:05",
                end_time="00:00:28:20",
                footprint=frozenset({(2, 5)}),
                footprint_size=1,
                within_slice_record_count=1,
                classification="rocky",
                reason="minimum_subdivision_size_reached",
                lasting_change_evidence_score=0.62,
                before_reference_activity=0.76,
                after_reference_activity=0.75,
                reference_windows_reliable=True,
            ),
            ClassifiedTimeSlice(
                slice_index=5,
                parent_union_index=4,
                slice_level=1,
                start_frame=860,
                end_frame=875,
                start_time="00:00:28:20",
                end_time="00:00:29:05",
                footprint=frozenset({(2, 6)}),
                footprint_size=1,
                within_slice_record_count=1,
                classification="rocky",
                reason="minimum_subdivision_size_reached",
                lasting_change_evidence_score=0.62,
                before_reference_activity=0.76,
                after_reference_activity=0.75,
                reference_windows_reliable=True,
            ),
        ]

        final_ranges = assemble_stage6_candidate_ranges(refined_slices)

        self.assertEqual(len(final_ranges), 1)
        self.assertEqual(final_ranges[0].start_frame, 800)
        self.assertEqual(final_ranges[0].source_classifications, ("rocky",))


    def test_stage6_drops_rocky_only_cluster_below_cluster_thresholds(self) -> None:
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
                footprint_size=1,
                within_slice_record_count=1,
                classification="rocky",
                reason="minimum_subdivision_size_reached",
                lasting_change_evidence_score=0.66,
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
                footprint_size=1,
                within_slice_record_count=1,
                classification="rocky",
                reason="minimum_subdivision_size_reached",
                lasting_change_evidence_score=0.67,
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
                footprint_size=1,
                within_slice_record_count=1,
                classification="rocky",
                reason="minimum_subdivision_size_reached",
                lasting_change_evidence_score=0.67,
                before_reference_activity=0.68,
                after_reference_activity=0.67,
                reference_windows_reliable=True,
            ),
        ]

        final_ranges = assemble_stage6_candidate_ranges(refined_slices)
        self.assertEqual(final_ranges, [])

    def test_stage6_merges_contiguous_valid_ranges_within_same_union(self) -> None:
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
                parent_union_index=2,
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
        self.assertFalse(final_ranges[0].includes_retained_undetermined)
        self.assertEqual(final_ranges[0].source_classifications, ("valid",))
if __name__ == "__main__":
    unittest.main()






















