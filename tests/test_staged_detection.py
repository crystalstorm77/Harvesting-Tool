from __future__ import annotations

# ============================================================
# SECTION A - Imports And Helpers
# ============================================================

import unittest
from collections import deque

import cv2
import numpy as np

from harvesting_tool.detection import ChapterRange, DetectorSettings, Timecode, extract_art_state_region
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
    classify_stage4_time_slices,
    classify_stage5_minimum_size_leaf,
    refine_stage5_sub_slices,
    screen_stage3_candidate_unions,
    select_stage3_art_state_reference_window,
)


def make_stage3_art_state_sample(
    frame_index: int,
    *,
    changed: bool = False,
) -> dict[str, object]:
    canvas_gray = np.zeros((120, 120), dtype=np.uint8)
    if changed:
        canvas_gray[40:70, 40:70] = 255
    return {
        'frame_index': frame_index,
        'canvas_gray': canvas_gray,
        'art_gray': extract_art_state_region(canvas_gray),
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
        union_footprint=frozenset({(4, 4), (4, 5), (5, 4), (5, 5)}) if union_footprint_size else frozenset(),
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
            union_footprint=frozenset({(4, 4), (4, 5), (5, 4), (5, 5)}),
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
            union_footprint=frozenset({(4, 4), (4, 5), (5, 4), (5, 5)}),
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
            union_footprint=frozenset({(4, 4), (4, 5), (5, 4), (5, 5)}),
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


    def test_stage3_art_state_prototype_survives_real_before_after_change(self) -> None:
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
            make_record(1, 304, movement_present=True, movement_strength_score=0.62, temporal_persistence_score=0.58, spatial_extent_score=0.5),
            make_record(2, 316, movement_present=True, movement_strength_score=0.66, temporal_persistence_score=0.60, spatial_extent_score=0.5),
        ]
        chapter_range = ChapterRange(
            start=Timecode.from_hhmmssff('00:00:00:00'),
            end=Timecode.from_hhmmssff('00:00:20:00'),
        )
        sampled_frames = [
            make_stage3_art_state_sample(285, changed=False),
            make_stage3_art_state_sample(290, changed=False),
            make_stage3_art_state_sample(336, changed=True),
            make_stage3_art_state_sample(340, changed=True),
            make_stage3_art_state_sample(348, changed=True),
            make_stage3_art_state_sample(352, changed=True),
        ]

        screened = screen_stage3_candidate_unions(
            [candidate_union],
            records,
            sampled_frames=sampled_frames,
            chapter_range=chapter_range,
            settings=make_settings(),
            use_art_state_prototype=True,
        )

        self.assertEqual(len(screened), 1)
        self.assertEqual(screened[0].screening_result, 'surviving', msg=repr(screened[0]))
        self.assertEqual(screened[0].reason, 'art_state_change_supported')
        self.assertEqual(screened[0].stage3_mode, 'art_state_prototype')
        self.assertGreater(screened[0].stage3_persistent_difference_score, 0.0)
        self.assertGreater(screened[0].stage3_footprint_support_score, 0.0)
        self.assertGreater(screened[0].stage3_after_window_persistence_score, 0.0)
        self.assertGreater(screened[0].stage3_reveal_window_hold_score, 0.0)

    def test_stage3_window_selector_prefers_stable_fallback_window(self) -> None:
        records = [
            make_record(1, 334, movement_present=True, movement_strength_score=0.18, temporal_persistence_score=0.16, spatial_extent_score=0.1),
            make_record(2, 336, movement_present=True, movement_strength_score=0.20, temporal_persistence_score=0.18, spatial_extent_score=0.1),
        ]
        sampled_frames = [
            make_stage3_art_state_sample(334, changed=False),
            make_stage3_art_state_sample(336, changed=True),
            make_stage3_art_state_sample(346, changed=True),
            make_stage3_art_state_sample(348, changed=True),
            make_stage3_art_state_sample(350, changed=True),
            make_stage3_art_state_sample(352, changed=True),
        ]

        selected_window, candidate_count = select_stage3_art_state_reference_window(
            search_start=333,
            search_end=353,
            union_anchor_frame=330,
            records=records,
            sampled_frames=sampled_frames,
            settings=make_settings(),
            cv2=cv2,
        )

        self.assertIsNotNone(selected_window)
        self.assertGreater(candidate_count, 1)
        self.assertEqual(selected_window['window_start'], 339)
        self.assertEqual(selected_window['tier'], 'local')

    def test_stage3_art_state_prototype_rejects_when_before_after_state_is_unchanged(self) -> None:
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
            make_record(1, 304, movement_present=True, movement_strength_score=0.62, temporal_persistence_score=0.58, spatial_extent_score=0.5),
            make_record(2, 316, movement_present=True, movement_strength_score=0.66, temporal_persistence_score=0.60, spatial_extent_score=0.5),
        ]
        chapter_range = ChapterRange(
            start=Timecode.from_hhmmssff('00:00:00:00'),
            end=Timecode.from_hhmmssff('00:00:20:00'),
        )
        sampled_frames = [
            make_stage3_art_state_sample(285, changed=False),
            make_stage3_art_state_sample(290, changed=False),
            make_stage3_art_state_sample(336, changed=False),
            make_stage3_art_state_sample(340, changed=False),
            make_stage3_art_state_sample(348, changed=False),
            make_stage3_art_state_sample(352, changed=False),
        ]

        screened = screen_stage3_candidate_unions(
            [candidate_union],
            records,
            sampled_frames=sampled_frames,
            chapter_range=chapter_range,
            settings=make_settings(),
            use_art_state_prototype=True,
        )

        self.assertEqual(screened[0].screening_result, 'rejected')
        self.assertEqual(screened[0].reason, 'weak_union_activity')
        self.assertEqual(screened[0].stage3_mode, 'art_state_prototype')

    def test_stage3_art_state_prototype_rejects_when_reveal_window_loses_post_state(self) -> None:
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
            make_record(1, 304, movement_present=True, movement_strength_score=0.62, temporal_persistence_score=0.58, spatial_extent_score=0.5),
            make_record(2, 316, movement_present=True, movement_strength_score=0.66, temporal_persistence_score=0.60, spatial_extent_score=0.5),
        ]
        chapter_range = ChapterRange(
            start=Timecode.from_hhmmssff('00:00:00:00'),
            end=Timecode.from_hhmmssff('00:00:20:00'),
        )
        sampled_frames = [
            make_stage3_art_state_sample(285, changed=False),
            make_stage3_art_state_sample(290, changed=False),
            make_stage3_art_state_sample(336, changed=True),
            make_stage3_art_state_sample(340, changed=True),
            make_stage3_art_state_sample(348, changed=False),
            make_stage3_art_state_sample(352, changed=False),
        ]

        screened = screen_stage3_candidate_unions(
            [candidate_union],
            records,
            sampled_frames=sampled_frames,
            chapter_range=chapter_range,
            settings=make_settings(),
            use_art_state_prototype=True,
        )

        self.assertEqual(screened[0].screening_result, 'surviving')
        self.assertEqual(screened[0].reason, 'art_state_change_supported')
        self.assertEqual(screened[0].stage3_mode, 'art_state_prototype')
        self.assertLess(screened[0].stage3_reveal_window_hold_score, 0.45)

    def test_stage3_art_state_prototype_rejects_when_footprint_support_is_too_small(self) -> None:
        candidate_union = CandidateUnion(
            union_index=1,
            start_frame=300,
            end_frame=330,
            start_time='00:00:10:00',
            end_time='00:00:11:00',
            member_movement_spans=(),
            union_footprint=frozenset({(row_index, column_index) for row_index in range(12) for column_index in range(12)}),
            union_footprint_size=144,
        )
        records = [
            make_record(1, 304, movement_present=True, movement_strength_score=0.62, temporal_persistence_score=0.58, spatial_extent_score=0.5),
            make_record(2, 316, movement_present=True, movement_strength_score=0.66, temporal_persistence_score=0.60, spatial_extent_score=0.5),
        ]
        chapter_range = ChapterRange(
            start=Timecode.from_hhmmssff('00:00:00:00'),
            end=Timecode.from_hhmmssff('00:00:20:00'),
        )
        def make_tiny_sample(frame_index: int, *, changed: bool) -> dict[str, object]:
            canvas_gray = np.zeros((120, 120), dtype=np.uint8)
            if changed:
                canvas_gray[58:62, 58:62] = 255
            return {
                'frame_index': frame_index,
                'canvas_gray': canvas_gray,
                'art_gray': extract_art_state_region(canvas_gray),
            }
        sampled_frames = [
            make_tiny_sample(285, changed=False),
            make_tiny_sample(290, changed=False),
            make_tiny_sample(336, changed=True),
            make_tiny_sample(340, changed=True),
            make_tiny_sample(348, changed=True),
            make_tiny_sample(352, changed=True),
        ]

        screened = screen_stage3_candidate_unions(
            [candidate_union],
            records,
            sampled_frames=sampled_frames,
            chapter_range=chapter_range,
            settings=make_settings(),
            use_art_state_prototype=True,
        )

        self.assertEqual(screened[0].screening_result, 'rejected')
        self.assertEqual(screened[0].reason, 'weak_union_activity')
        self.assertLess(screened[0].stage3_footprint_support_score, 0.10)

    def test_stage3_art_state_prototype_rejects_small_fallback_post_reference_without_reveal(self) -> None:
        candidate_union = CandidateUnion(
            union_index=1,
            start_frame=300,
            end_frame=330,
            start_time='00:00:10:00',
            end_time='00:00:11:00',
            member_movement_spans=(),
            union_footprint=frozenset({(1, 8), (1, 9), (2, 8), (2, 9), (3, 6), (3, 7), (3, 8), (3, 9), (4, 6), (4, 7)}),
            union_footprint_size=10,
        )
        records = [
            make_record(1, 304, movement_present=True, movement_strength_score=0.62, temporal_persistence_score=0.58, spatial_extent_score=0.5),
            make_record(2, 316, movement_present=True, movement_strength_score=0.66, temporal_persistence_score=0.60, spatial_extent_score=0.5),
        ]
        chapter_range = ChapterRange(
            start=Timecode.from_hhmmssff('00:00:00:00'),
            end=Timecode.from_hhmmssff('00:00:12:16'),
        )
        sampled_frames = [
            make_stage3_art_state_sample(285, changed=False),
            make_stage3_art_state_sample(290, changed=False),
            make_stage3_art_state_sample(370, changed=True),
            make_stage3_art_state_sample(374, changed=True),
        ]

        screened = screen_stage3_candidate_unions(
            [candidate_union],
            records,
            sampled_frames=sampled_frames,
            chapter_range=chapter_range,
            settings=make_settings(),
            use_art_state_prototype=True,
        )

        self.assertEqual(screened[0].screening_result, 'rejected')
        self.assertEqual(screened[0].reason, 'fallback_post_reference_too_small')
        self.assertEqual(screened[0].stage3_after_window_tier, 'fallback')
        self.assertEqual(screened[0].stage3_reveal_sample_count, 0)
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

    def test_stage4_keeps_moderately_strong_active_reference_slice_undetermined_in_long_strong_union(self) -> None:
        screened_union = make_screened_union(start_frame=300, end_frame=960)
        screened_union = screened_union.__class__(
            candidate_union=screened_union.candidate_union,
            screening_result=screened_union.screening_result,
            surviving=screened_union.surviving,
            provisional_survival=screened_union.provisional_survival,
            reason=screened_union.reason,
            within_union_record_count=screened_union.within_union_record_count,
            before_record_count=screened_union.before_record_count,
            after_record_count=screened_union.after_record_count,
            mean_movement_strength=screened_union.mean_movement_strength,
            mean_temporal_persistence=screened_union.mean_temporal_persistence,
            mean_spatial_extent=screened_union.mean_spatial_extent,
            lasting_change_evidence_score=0.73,
            before_reference_activity=screened_union.before_reference_activity,
            after_reference_activity=screened_union.after_reference_activity,
            reference_windows_reliable=screened_union.reference_windows_reliable,
        )
        records = [
            make_record(1, 304, movement_present=True, movement_strength_score=0.76, temporal_persistence_score=0.74, spatial_extent_score=0.72, touched_grid_coordinates=((2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4))),
            make_record(2, 332, movement_present=True, movement_strength_score=0.75, temporal_persistence_score=0.73, spatial_extent_score=0.72, touched_grid_coordinates=((3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4))),
            make_record(3, 292, movement_present=True, movement_strength_score=0.76, temporal_persistence_score=0.74, spatial_extent_score=0.62),
            make_record(4, 298, movement_present=True, movement_strength_score=0.75, temporal_persistence_score=0.73, spatial_extent_score=0.62),
            make_record(5, 632, movement_present=True, movement_strength_score=0.75, temporal_persistence_score=0.73, spatial_extent_score=0.62),
            make_record(6, 638, movement_present=True, movement_strength_score=0.76, temporal_persistence_score=0.74, spatial_extent_score=0.62),
        ]

        slices = classify_stage4_time_slices([screened_union], records)

        self.assertEqual(len(slices), 2)
        self.assertEqual(slices[0].classification, "undetermined")
        self.assertEqual(slices[0].reason, "reference_windows_too_active")

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

        self.assertTrue(any(slice_info.classification == "boundary" and slice_info.reason == "minimum_subdivision_size_reached" for slice_info in refined))
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
                footprint_size=8,
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

        self.assertEqual(rescued.classification, "boundary")
        self.assertEqual(rescued.reason, "active_reference_minimum_size_reached")

    def test_stage5_rescues_art_state_supported_minimum_size_leaf(self) -> None:
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
                footprint_size=9,
                within_slice_record_count=2,
                classification="invalid",
                reason="reference_windows_too_active",
                lasting_change_evidence_score=0.59,
                before_reference_activity=0.76,
                after_reference_activity=0.75,
                reference_windows_reliable=True,
            ),
            allow_art_state_supported_rescue=True,
        )

        self.assertEqual(rescued.classification, "boundary")
        self.assertEqual(rescued.reason, "art_state_supported_minimum_size_reached")

    def test_stage5_promotes_strong_minimum_size_leaf_to_valid_anchor(self) -> None:
        promoted = classify_stage5_minimum_size_leaf(
            ClassifiedTimeSlice(
                slice_index=1,
                parent_union_index=1,
                slice_level=3,
                start_frame=300,
                end_frame=315,
                start_time="00:00:10:00",
                end_time="00:00:10:15",
                footprint=frozenset({(2, 2), (2, 3)}),
                footprint_size=8,
                within_slice_record_count=2,
                classification="undetermined",
                reason="mixed_slice_evidence",
                lasting_change_evidence_score=0.60,
                before_reference_activity=0.76,
                after_reference_activity=0.75,
                reference_windows_reliable=True,
            ),
            allow_valid_anchor_promotion=True,
        )

        self.assertEqual(promoted.classification, "valid")
        self.assertEqual(promoted.reason, "slice_activity_supported")

    def test_stage5_does_not_promote_small_minimum_size_leaf_to_valid_anchor(self) -> None:
        retained = classify_stage5_minimum_size_leaf(
            ClassifiedTimeSlice(
                slice_index=1,
                parent_union_index=1,
                slice_level=3,
                start_frame=300,
                end_frame=315,
                start_time="00:00:10:00",
                end_time="00:00:10:15",
                footprint=frozenset({(2, 2), (2, 3)}),
                footprint_size=7,
                within_slice_record_count=2,
                classification="undetermined",
                reason="mixed_slice_evidence",
                lasting_change_evidence_score=0.60,
                before_reference_activity=0.76,
                after_reference_activity=0.75,
                reference_windows_reliable=True,
            ),
            allow_valid_anchor_promotion=True,
        )

        self.assertEqual(retained.classification, "boundary")
        self.assertEqual(retained.reason, "minimum_subdivision_size_reached")

    def test_stage5_does_not_promote_local_strong_leaf_without_coherent_sibling_support(self) -> None:
        screened_union = make_screened_union(start_frame=300, end_frame=960)
        screened_union = screened_union.__class__(
            candidate_union=screened_union.candidate_union,
            screening_result=screened_union.screening_result,
            surviving=screened_union.surviving,
            provisional_survival=screened_union.provisional_survival,
            reason=screened_union.reason,
            within_union_record_count=screened_union.within_union_record_count,
            before_record_count=screened_union.before_record_count,
            after_record_count=screened_union.after_record_count,
            mean_movement_strength=screened_union.mean_movement_strength,
            mean_temporal_persistence=screened_union.mean_temporal_persistence,
            mean_spatial_extent=screened_union.mean_spatial_extent,
            lasting_change_evidence_score=0.73,
            before_reference_activity=screened_union.before_reference_activity,
            after_reference_activity=screened_union.after_reference_activity,
            reference_windows_reliable=screened_union.reference_windows_reliable,
        )
        existing_valid_slice = ClassifiedTimeSlice(
            slice_index=1,
            parent_union_index=screened_union.candidate_union.union_index,
            slice_level=0,
            start_frame=900,
            end_frame=930,
            start_time="00:00:30:00",
            end_time="00:00:31:00",
            footprint=frozenset({(5, 5), (5, 6), (6, 5), (6, 6)}),
            footprint_size=8,
            within_slice_record_count=2,
            classification="valid",
            reason="slice_activity_supported",
            lasting_change_evidence_score=0.82,
            before_reference_activity=0.02,
            after_reference_activity=0.02,
            reference_windows_reliable=True,
        )
        local_undetermined_slice = ClassifiedTimeSlice(
            slice_index=2,
            parent_union_index=screened_union.candidate_union.union_index,
            slice_level=0,
            start_frame=300,
            end_frame=315,
            start_time="00:00:10:00",
            end_time="00:00:10:15",
            footprint=frozenset({(2, 2), (2, 3)}),
            footprint_size=8,
            within_slice_record_count=2,
            classification="undetermined",
            reason="mixed_reference_activity",
            lasting_change_evidence_score=0.60,
            before_reference_activity=0.22,
            after_reference_activity=0.30,
            reference_windows_reliable=True,
        )

        refined = refine_stage5_sub_slices(
            [screened_union],
            [existing_valid_slice, local_undetermined_slice],
            [],
            minimum_subdivision_frames=15,
        )

        promoted = next(
            slice_info
            for slice_info in refined
            if slice_info.start_frame == 300 and slice_info.end_frame == 315
        )
        self.assertEqual(promoted.classification, "boundary")
        self.assertEqual(promoted.reason, "minimum_subdivision_size_reached")

    def test_stage5_rescues_moderate_undetermined_leaf_in_long_strong_union(self) -> None:
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
                footprint_size=8,
                within_slice_record_count=2,
                classification="undetermined",
                reason="reference_windows_too_active",
                lasting_change_evidence_score=0.59,
                before_reference_activity=0.76,
                after_reference_activity=0.75,
                reference_windows_reliable=True,
            ),
            allow_long_strong_union_rocky_rescue=True,
        )

        self.assertEqual(rescued.classification, "boundary")
        self.assertEqual(rescued.reason, "long_strong_union_minimum_size_reached")


    def test_stage5_rescues_invalid_leaf_when_parent_activity_is_strong(self) -> None:
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
                footprint_size=8,
                within_slice_record_count=2,
                classification="invalid",
                reason="reference_windows_too_active",
                lasting_change_evidence_score=0.59,
                before_reference_activity=0.76,
                after_reference_activity=0.75,
                reference_windows_reliable=True,
            ),
            allow_high_parent_activity_rescue=True,
        )

        self.assertEqual(rescued.classification, "boundary")
        self.assertEqual(rescued.reason, "high_parent_activity_minimum_size_reached")

    def test_stage5_rescues_reference_unreliable_undetermined_leaf(self) -> None:
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
                footprint_size=8,
                within_slice_record_count=2,
                classification="undetermined",
                reason="minimum_subdivision_size_reached",
                lasting_change_evidence_score=0.61,
                before_reference_activity=0.76,
                after_reference_activity=0.75,
                reference_windows_reliable=False,
            ),
            allow_reference_unreliable_rescue=True,
        )

        self.assertEqual(rescued.classification, "boundary")
        self.assertEqual(rescued.reason, "reference_unreliable_minimum_size_reached")

    def test_stage5_rescues_structural_gap_invalid_leaf(self) -> None:
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
                footprint_size=8,
                within_slice_record_count=2,
                classification="invalid",
                reason="reference_windows_too_active",
                lasting_change_evidence_score=0.58,
                before_reference_activity=0.76,
                after_reference_activity=0.75,
                reference_windows_reliable=True,
            ),
            allow_structural_gap_rescue=True,
        )

        self.assertEqual(rescued.classification, "boundary")
        self.assertEqual(rescued.reason, "structural_gap_minimum_size_reached")

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
        self.assertTrue(final_ranges[0].includes_retained_undetermined)
        self.assertEqual(final_ranges[0].retained_undetermined_count, 1)
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
                reason="reference_unreliable_minimum_size_reached",
                lasting_change_evidence_score=0.69,
                before_reference_activity=0.72,
                after_reference_activity=0.68,
                reference_windows_reliable=True,
            ),
            ClassifiedTimeSlice(
                slice_index=2,
                parent_union_index=1,
                slice_level=1,
                start_frame=338,
                end_frame=353,
                start_time="00:00:11:08",
                end_time="00:00:11:23",
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
        self.assertEqual(final_ranges[0].end_frame, 353)
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

    def test_stage6_keeps_short_abrupt_canvas_change_boundary_cluster(self) -> None:
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

        self.assertEqual(len(final_ranges), 1)
        self.assertEqual(final_ranges[0].start_frame, 900)
        self.assertEqual(final_ranges[0].end_frame, 916)
        self.assertEqual(final_ranges[0].source_classifications, ("boundary",))

    def test_stage6_does_not_merge_low_overlap_boundary_slice_into_valid_boundary(self) -> None:
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
        self.assertEqual(final_ranges[0].start_frame, 1024)
        self.assertEqual(final_ranges[0].source_classifications, ("valid",))

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





