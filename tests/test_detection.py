# ============================================================
# SECTION A - Imports And Helpers
# ============================================================

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harvesting_tool.detection import (
    DetectorSettings,
    Timecode,
    build_candidate_clips,
    build_cut_list_payload,
    format_cut_list_text,
    parse_chapter_range,
    write_cut_lists,
)


def make_settings() -> DetectorSettings:
    return DetectorSettings(
        lead_in=Timecode.from_seconds_and_frames(2, 10),
        tail_after=Timecode.from_seconds_and_frames(1, 5),
        min_harvest=Timecode.from_hhmmssff("00:00:10:00"),
        max_harvest=Timecode.from_hhmmssff("00:01:00:00"),
        max_clip_length=Timecode.from_hhmmssff("00:00:12:00"),
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
        self.assertEqual(payload["actual"]["clip_count"], 1)
        self.assertEqual(payload["clips"][0]["clip_start"], "00:00:07:20")


# ============================================================
# SECTION C - Candidate Clip Construction Tests
# ============================================================

class CandidateClipTests(unittest.TestCase):
    def test_long_continuous_activity_is_split_into_back_to_back_clips(self) -> None:
        settings = make_settings()
        chapter = parse_chapter_range("00:00:00:00", "00:10:00:00")
        clips = build_candidate_clips(
            "sample.mp4",
            chapter,
            bursts=[(300, 1200)],
            settings=settings,
        )

        self.assertEqual(len(clips), 4)
        self.assertEqual(clips[0].activity_start.to_hhmmssff(), "00:00:10:00")
        self.assertEqual(clips[0].activity_end.to_hhmmssff(), "00:00:18:15")
        self.assertEqual(clips[1].activity_start.to_hhmmssff(), "00:00:18:15")
        self.assertEqual(clips[1].activity_end.to_hhmmssff(), "00:00:27:00")
        self.assertEqual(clips[2].activity_start.to_hhmmssff(), "00:00:27:00")
        self.assertEqual(clips[2].activity_end.to_hhmmssff(), "00:00:35:15")
        self.assertEqual(clips[3].activity_start.to_hhmmssff(), "00:00:35:15")
        self.assertEqual(clips[3].activity_end.to_hhmmssff(), "00:00:40:00")

    def test_harvest_maximum_caps_total_returned_duration_after_full_scan(self) -> None:
        settings = DetectorSettings(
            lead_in=Timecode.from_seconds_and_frames(1, 0),
            tail_after=Timecode.from_seconds_and_frames(0, 15),
            min_harvest=Timecode.from_hhmmssff("00:00:05:00"),
            max_harvest=Timecode.from_hhmmssff("00:00:12:00"),
            max_clip_length=Timecode.from_hhmmssff("00:00:08:00"),
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
        self.assertEqual(clips[0].clip_start.to_hhmmssff(), "00:00:09:00")
        total_frames = sum(clip.duration.total_frames for clip in clips)
        self.assertEqual(Timecode(total_frames=total_frames).to_hhmmssff(), "00:00:12:00")


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
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["clips"][0]["clip_end"], "00:00:19:05")


if __name__ == "__main__":
    unittest.main()
