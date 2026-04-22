# ============================================================
# SECTION A - Imports And Helpers
# ============================================================

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from harvesting_tool.detection import (
    ART_STATE_BASELINE_MAX_SAMPLES,
    ART_STATE_REVEAL_WINDOW_FRAMES,
    VALIDATION_MERGE_GAP_FRAMES,
    build_scan_resolution_display_labels,
    build_scan_resolution_metadata,
    compute_scaled_dimensions,
    DetectionDebugBundle,
    DetectorSettings,
    SampleDebugRow,
    Timecode,
    backtrack_event_start,
    build_art_state_windows,
    build_candidate_unions,
    build_movement_spans,
    refine_surviving_unions,
    assemble_final_activity_ranges,
    build_reveal_window,
    build_window_footprint_mask,
    find_effective_update_end,
    build_candidate_clips,
    build_cut_list_payload,
    classify_activity_signal,
    compute_enter_ratio_threshold,
    compute_remain_ratio_threshold,
    emit_scan_progress,
    format_cut_list_text,
    has_localized_activity_support,
    is_weak_art_change_signal,
    merge_activity_bursts,
    merge_validated_subranges,
    find_onset_recovery_start,
    should_continue_refined_run,
    parse_chapter_range,
    select_representative_samples,
    should_enter_active_state,
    should_remain_active_state,
    write_cut_lists,
    write_debug_artifacts,
)


def make_settings() -> DetectorSettings:
    return DetectorSettings(
        lead_in=Timecode.from_seconds_and_frames(0, 2),
        tail_after=Timecode.from_seconds_and_frames(0, 4),
        min_harvest=Timecode.from_hhmmssff("00:00:10:00"),
        max_harvest=Timecode.from_hhmmssff("00:01:00:00"),
        min_clip_length=Timecode.from_hhmmssff("00:00:00:15"),
        max_clip_length=Timecode.from_hhmmssff("00:00:07:00"),
        pause_threshold=Timecode.from_hhmmssff("00:00:05:00"),
        min_burst_length=Timecode.from_hhmmssff("00:00:00:10"),
    )
# ============================================================
# SECTION B - Foundation Tests
# ============================================================

class TimecodeAndOutputTests(unittest.TestCase):
    def test_parse_chapter_range_uses_30_fps_timecodes(self) -> None:
        chapter = parse_chapter_range("00:01:02:15", "00:02:00:00")
        self.assertEqual(chapter.start.total_frames, ((62 * 30) + 15))
        self.assertEqual(chapter.end.total_frames, 120 * 30)

    def test_text_and_json_output_include_expected_fields(self) -> None:
        settings = make_settings()
        chapter = parse_chapter_range("00:00:00:00", "00:05:00:00")
        clips = build_candidate_clips(
            "sample.mp4",
            chapter,
            bursts=[(300, 540)],
            settings=settings,
        )

        text_output = format_cut_list_text(clips)
        payload = build_cut_list_payload(clips, settings)

        self.assertIn("Clip 1", text_output)
        self.assertIn("sample.mp4", text_output)
        self.assertEqual(payload["frame_rate"], 30)
        self.assertEqual(payload["actual"]["clip_count"], 2)
        self.assertEqual(payload["clips"][0]["clip_start"], "00:00:09:28")
        self.assertEqual(payload["requested"]["min_clip_length"], "00:00:00:15")

    def test_progress_emits_rounded_checkpoints(self) -> None:
        chapter = parse_chapter_range("00:00:00:00", "00:00:10:00")
        emitted: list[int] = []
        last_percent = -5

        last_percent = emit_scan_progress(15, chapter, emitted.append, last_percent)
        last_percent = emit_scan_progress(60, chapter, emitted.append, last_percent)
        last_percent = emit_scan_progress(165, chapter, emitted.append, last_percent)

        self.assertEqual(last_percent, 55)
        self.assertEqual(emitted, [5, 20, 55])

    def test_write_debug_artifacts_creates_expected_files(self) -> None:
        debug_bundle = DetectionDebugBundle(
            sampled_frames=[
                SampleDebugRow(
                    frame_index=300,
                    timecode="00:00:10:00",
                    adjacent_change_score=0.1,
                    persistent_change_score=0.05,
                    adjacent_blocks=4,
                    persistent_blocks=3,
                    trail_blocks=5,
                    trail_excess_blocks=2,
                    trail_persistent_ratio=1.666667,
                    locality_score=0.02,
                    global_change_score=0.1,
                    enter_active=True,
                    remain_active=False,
                    active_state=True,
                    micro_event_marker="start",
                    notes="persistent_blocks=3",
                )
            ],
            micro_events=[{"micro_event_index": 1, "start": "00:00:10:00", "end": "00:00:10:10"}],
            merged_bursts=[{"burst_index": 1, "start": "00:00:10:00", "end": "00:00:10:10"}],
            final_candidate_clips=[{"clip_index": 1, "clip_start": "00:00:09:28", "clip_end": "00:00:10:14"}],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_debug_artifacts(Path(tmpdir) / "debug_run", debug_bundle)
            self.assertTrue(paths["frames_csv"].exists())
            self.assertTrue(paths["micro_events_json"].exists())
            self.assertTrue(paths["bursts_json"].exists())
            self.assertTrue(paths["candidate_clips_json"].exists())
            self.assertIn("micro_event_marker", paths["frames_csv"].read_text(encoding="utf-8"))

    def test_scan_resolution_display_labels_match_expected_sizes(self) -> None:
        labels = build_scan_resolution_display_labels(1920, 1080)

        self.assertEqual(labels["full"], "1920x1080")
        self.assertEqual(labels["half"], "960x540")
        self.assertEqual(labels["quarter"], "480x270")
        self.assertEqual(labels["eighth"], "240x135")

    def test_scan_resolution_metadata_tracks_selected_frame_and_canvas_sizes(self) -> None:
        settings = make_settings()
        settings = DetectorSettings(**{**settings.__dict__, "scan_resolution": "half"})

        metadata = build_scan_resolution_metadata(3840, 2160, settings)

        self.assertEqual(metadata["mode"], "half")
        self.assertEqual(metadata["selected_frame_width"], 1920)
        self.assertEqual(metadata["selected_frame_height"], 1080)
        self.assertGreater(int(metadata["source_canvas_width"]), int(metadata["analysis_canvas_width"]))

    def test_compute_scaled_dimensions_uses_floor_division_with_minimum_one(self) -> None:
        self.assertEqual(compute_scaled_dimensions(1919, 1079, "half"), (959, 539))
        self.assertEqual(compute_scaled_dimensions(3, 3, "eighth"), (1, 1))


# ============================================================
# SECTION C - Candidate Clip Construction Tests
# ============================================================

class CandidateClipTests(unittest.TestCase):
    def test_build_movement_spans_discards_raw_events_below_minimum_length(self) -> None:
        settings = make_settings()
        movement_spans = build_movement_spans(
            [(300, 306), (360, 390)],
            settings,
        )

        self.assertEqual(movement_spans, [(360, 390)])

    def test_build_candidate_unions_merges_nearby_movement_spans_with_validation_gap(self) -> None:
        candidate_unions = build_candidate_unions([(300, 330), (345, 375), (480, 510)])
        self.assertEqual(candidate_unions, [(300, 375), (480, 510)])

    def test_refine_surviving_unions_keeps_only_screened_survivors(self) -> None:
        screened_candidate_unions = [
            {
                'validated': False,
                'surviving_ranges_frames': [(300, 360)],
                'candidate_union_index': 1,
            },
            {
                'validated': True,
                'surviving_ranges_frames': [(420, 450), (480, 510)],
                'candidate_union_index': 2,
            },
        ]

        surviving_union_ranges = refine_surviving_unions(screened_candidate_unions)
        self.assertEqual(surviving_union_ranges, [(420, 450), (480, 510)])

    def test_assemble_final_activity_ranges_applies_post_refinement_merge(self) -> None:
        final_activity_ranges = assemble_final_activity_ranges([(300, 330), (345, 375), (480, 510)])
        self.assertEqual(final_activity_ranges, [(300, 375), (480, 510)])
    def test_nearby_bursts_merge_when_gap_is_within_pause_threshold(self) -> None:
        settings = make_settings()
        merged = merge_activity_bursts([(300, 330), (360, 390), (600, 630)], settings.pause_threshold)
        self.assertEqual(merged, [(300, 390), (600, 630)])

    def test_short_detected_activity_expands_to_minimum_clip_length(self) -> None:
        settings = make_settings()
        chapter = parse_chapter_range("00:00:00:00", "00:01:00:00")
        clips = build_candidate_clips(
            "sample.mp4",
            chapter,
            bursts=[(300, 312)],
            settings=settings,
        )

        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0].duration.to_hhmmssff(), "00:00:00:18")

    def test_long_continuous_activity_is_split_into_back_to_back_clips(self) -> None:
        settings = make_settings()
        chapter = parse_chapter_range("00:00:00:00", "00:10:00:00")
        clips = build_candidate_clips(
            "sample.mp4",
            chapter,
            bursts=[(300, 1200)],
            settings=settings,
        )

        self.assertEqual(len(clips), 5)
        self.assertEqual(clips[0].activity_start.to_hhmmssff(), "00:00:10:00")
        self.assertEqual(clips[0].activity_end.to_hhmmssff(), "00:00:16:24")
        self.assertEqual(clips[1].activity_start.to_hhmmssff(), "00:00:16:24")
        self.assertEqual(clips[1].activity_end.to_hhmmssff(), "00:00:23:18")
        self.assertEqual(clips[4].activity_start.to_hhmmssff(), "00:00:37:06")
        self.assertEqual(clips[4].activity_end.to_hhmmssff(), "00:00:40:00")

    def test_harvest_maximum_caps_total_returned_duration_after_full_scan(self) -> None:
        settings = DetectorSettings(
            lead_in=Timecode.from_seconds_and_frames(0, 2),
            tail_after=Timecode.from_seconds_and_frames(0, 4),
            min_harvest=Timecode.from_hhmmssff("00:00:05:00"),
            max_harvest=Timecode.from_hhmmssff("00:00:12:00"),
            min_clip_length=Timecode.from_hhmmssff("00:00:00:15"),
            max_clip_length=Timecode.from_hhmmssff("00:00:07:00"),
            pause_threshold=Timecode.from_hhmmssff("00:00:05:00"),
            min_burst_length=Timecode.from_hhmmssff("00:00:00:10"),
        )
        chapter = parse_chapter_range("00:00:00:00", "00:10:00:00")
        clips = build_candidate_clips(
            "sample.mp4",
            chapter,
            bursts=[(300, 420), (900, 1050)],
            settings=settings,
        )

        self.assertEqual(len(clips), 2)
        total_frames = sum(clip.duration.total_frames for clip in clips)
        self.assertEqual(Timecode(total_frames=total_frames).to_hhmmssff(), "00:00:09:12")

    def test_candidate_clip_clamps_activity_end_and_tail_after_at_chapter_end(self) -> None:
        settings = make_settings()
        chapter = parse_chapter_range("00:00:00:00", "00:00:10:00")
        clips = build_candidate_clips(
            "sample.mp4",
            chapter,
            bursts=[(295, 305)],
            settings=settings,
            trust_burst_boundaries=True,
        )

        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0].activity_end.to_hhmmssff(), "00:00:10:00")
        self.assertEqual(clips[0].clip_end.to_hhmmssff(), "00:00:10:00")
        self.assertEqual(clips[0].tail_after.to_seconds_frames(), {"seconds": 0, "frames": 0})

    def test_trusted_burst_boundaries_skip_editorial_remerge(self) -> None:
        settings = make_settings()
        chapter = parse_chapter_range("00:00:00:00", "00:01:00:00")
        clips = build_candidate_clips(
            "sample.mp4",
            chapter,
            bursts=[(300, 360), (420, 480)],
            settings=settings,
            trust_burst_boundaries=True,
        )

        self.assertEqual(len(clips), 2)
        self.assertEqual(clips[0].activity_start.to_hhmmssff(), "00:00:10:00")
        self.assertEqual(clips[0].activity_end.to_hhmmssff(), "00:00:12:00")
        self.assertEqual(clips[1].activity_start.to_hhmmssff(), "00:00:14:00")
        self.assertEqual(clips[1].activity_end.to_hhmmssff(), "00:00:16:00")
    def test_thresholds_allow_weaker_remain_than_enter(self) -> None:
        settings = make_settings()
        self.assertGreater(compute_enter_ratio_threshold(settings), compute_remain_ratio_threshold(settings))
        self.assertTrue(should_enter_active_state(0.006, 0.0004, 6, 3, 4, settings))
        self.assertFalse(should_enter_active_state(0.006, 0.00002, 6, 1, 0, settings))
        self.assertFalse(should_remain_active_state(0.004, 0.00012, 4, 1, 0, settings))
        self.assertTrue(should_remain_active_state(0.004, 0.00012, 5, 2, 3, settings))

    def test_backtrack_finds_earliest_recent_weak_signal(self) -> None:
        recent_signals = [(9476, False), (9478, True), (9480, True), (9482, True), (9484, True)]
        self.assertEqual(backtrack_event_start(recent_signals, 9484), 9478)

    def test_weak_signal_detection_matches_expected_low_level_onset(self) -> None:
        settings = make_settings()
        self.assertTrue(is_weak_art_change_signal(0.001, 0.00008, 2, 1, settings))
        self.assertFalse(is_weak_art_change_signal(0.0, 0.0, 0, 0, settings))

    def test_activity_classifier_prefers_persistent_changes_over_transient_motion(self) -> None:
        settings = make_settings()
        self.assertTrue(classify_activity_signal(0.007, 0.0004, 6, 3, 4, settings))
        self.assertFalse(classify_activity_signal(0.009, 0.00002, 6, 0, 0, settings))

    def test_select_representative_samples_caps_large_windows(self) -> None:
        samples = [{'frame_index': index} for index in range(40)]
        selected = select_representative_samples(samples)

        self.assertEqual(len(selected), ART_STATE_BASELINE_MAX_SAMPLES)
        self.assertEqual(selected[0]['frame_index'], 0)
        self.assertEqual(selected[-1]['frame_index'], 39)

    def test_art_state_windows_are_bounded_by_neighboring_bursts(self) -> None:
        settings = make_settings()
        chapter = parse_chapter_range("00:00:00:00", "00:02:00:00")
        merged_bursts = [(300, 360), (600, 660), (900, 960)]

        (pre_start, pre_end), (post_start, post_end) = build_art_state_windows(
            burst_index=1,
            merged_bursts=merged_bursts,
            effective_update_end=660,
            chapter_range=chapter,
            settings=settings,
        )

        self.assertEqual(pre_start, 360)
        self.assertEqual(pre_end, 600)
        self.assertEqual(post_start, 663)
        self.assertEqual(post_end, 900)

    def test_reveal_window_uses_later_samples_after_post_window(self) -> None:
        chapter = parse_chapter_range("00:00:00:00", "00:03:00:00")
        merged_bursts = [(300, 360), (600, 660), (900, 960), (1200, 1260)]

        reveal_start, reveal_end = build_reveal_window(
            burst_index=1,
            merged_bursts=merged_bursts,
            chapter_range=chapter,
            effective_update_end=900,
            raw_burst_end=930,
        )

        self.assertEqual(reveal_start, 930)
        self.assertEqual(reveal_end, min(1200, 930 + ART_STATE_REVEAL_WINDOW_FRAMES))


    def test_merge_validated_subranges_combines_overlapping_windows(self) -> None:
        merged = merge_validated_subranges([(300, 330), (315, 345), (390, 420)], 15)
        self.assertEqual(merged, [(300, 345), (390, 420)])

    def test_find_onset_recovery_start_stops_before_weak_footprint_windows(self) -> None:
        window_evaluations = [
            {'start': 300, 'activity_support': True, 'overlay_like': False, 'footprint_art_state_blocks': 1},
            {'start': 330, 'activity_support': True, 'overlay_like': False, 'footprint_art_state_blocks': 4},
            {'start': 360, 'activity_support': True, 'overlay_like': False, 'footprint_art_state_blocks': 6},
        ]

        recovery_start = find_onset_recovery_start(window_evaluations, 2)
        self.assertEqual(recovery_start, 330)

    def test_should_continue_refined_run_requires_real_footprint_persistence_without_anchor(self) -> None:
        weak_window = {
            'anchor_valid': False,
            'activity_support': True,
            'footprint_art_state_blocks': 2,
            'footprint_art_state_ratio': 0.001,
        }
        strong_window = {
            'anchor_valid': False,
            'activity_support': True,
            'footprint_art_state_blocks': 6,
            'footprint_art_state_ratio': 0.004,
        }

        self.assertFalse(should_continue_refined_run(weak_window, 0.015))
        self.assertTrue(should_continue_refined_run(strong_window, 0.015))

    def test_localized_activity_support_requires_real_short_term_signal(self) -> None:
        settings = make_settings()
        signal_rows = [
            {
                'frame_index': 300,
                'adjacent_ratio': 0.006,
                'persistent_ratio': 0.0004,
                'adjacent_blocks': 6,
                'persistent_blocks': 3,
                'trail_excess_blocks': 4,
            },
            {
                'frame_index': 330,
                'adjacent_ratio': 0.003,
                'persistent_ratio': 0.002,
                'adjacent_blocks': 4,
                'persistent_blocks': 4,
                'trail_excess_blocks': 0,
            },
        ]

        self.assertTrue(has_localized_activity_support(300, 315, signal_rows, settings))
        self.assertFalse(has_localized_activity_support(345, 360, signal_rows, settings))

    def test_window_footprint_mask_tracks_touched_blocks(self) -> None:
        settings = make_settings()
        first = np.zeros((120, 120), dtype=np.uint8)
        second = first.copy()
        second[20:40, 30:50] = 255
        window_samples = [
            {'art_gray': first},
            {'art_gray': second},
        ]

        footprint_mask = build_window_footprint_mask(window_samples, settings, cv2)

        self.assertIsNotNone(footprint_mask)
        self.assertGreaterEqual(int((footprint_mask > 0).sum()), 100)

    def test_post_validation_merge_uses_tighter_gap_than_editorial_pause_threshold(self) -> None:
        merged = merge_activity_bursts(
            [(300, 360), (480, 540)],
            Timecode(total_frames=VALIDATION_MERGE_GAP_FRAMES),
        )
        self.assertEqual(merged, [(300, 360), (480, 540)])

    def test_post_validation_merge_still_combines_small_adjacent_ranges(self) -> None:
        merged = merge_activity_bursts(
            [(300, 330), (345, 375)],
            Timecode(total_frames=VALIDATION_MERGE_GAP_FRAMES),
        )
        self.assertEqual(merged, [(300, 375)])
    def test_effective_update_end_trims_idle_hold_tail(self) -> None:
        settings = make_settings()
        signal_rows = [
            {'frame_index': 300, 'adjacent_ratio': 0.01, 'persistent_ratio': 0.004},
            {'frame_index': 303, 'adjacent_ratio': 0.009, 'persistent_ratio': 0.003},
            {'frame_index': 306, 'adjacent_ratio': 0.0, 'persistent_ratio': 0.0},
            {'frame_index': 309, 'adjacent_ratio': 0.0, 'persistent_ratio': 0.0},
        ]

        effective_end = find_effective_update_end(
            burst_start=300,
            burst_end=312,
            signal_rows=signal_rows,
            settings=settings,
        )

        self.assertEqual(effective_end, 306)
# ============================================================
# SECTION D - File Writing Tests
# ============================================================

class CutListWritingTests(unittest.TestCase):
    def test_write_cut_lists_creates_text_and_json_files(self) -> None:
        settings = make_settings()
        chapter = parse_chapter_range("00:00:00:00", "00:05:00:00")
        clips = build_candidate_clips(
            "sample.mp4",
            chapter,
            bursts=[(300, 540)],
            settings=settings,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            stem = Path(tmpdir) / "cut_list"
            text_path, json_path = write_cut_lists(stem, clips, settings)

            self.assertTrue(text_path.exists())
            self.assertTrue(json_path.exists())
            self.assertEqual(text_path.name, "cut_list - Final Harvested Clips.txt")
            self.assertEqual(json_path.name, "cut_list - Final Harvested Clips - Machine Readable.json")
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["clips"][0]["clip_end"], "00:00:16:28")


if __name__ == "__main__":
    unittest.main()
