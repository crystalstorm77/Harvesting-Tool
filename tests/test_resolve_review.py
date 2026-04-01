# ============================================================
# SECTION A - Imports And Helpers
# ============================================================

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from harvesting_tool.detection import CandidateClip, Timecode, parse_chapter_range
from harvesting_tool.resolve_review import (
    ResolveClipPlacement,
    build_gap_preserved_placements,
    build_packed_placements,
    build_source_reference_placement,
    candidate_clip_from_dict,
    load_review_payload,
)



def make_clip(index: int, start_tc: str, end_tc: str) -> CandidateClip:
    chapter = parse_chapter_range("00:00:00:00", "00:10:00:00")
    return CandidateClip(
        clip_index=index,
        source_path="sample.mp4",
        chapter_start=chapter.start,
        chapter_end=chapter.end,
        activity_start=Timecode.from_hhmmssff(start_tc),
        activity_end=Timecode.from_hhmmssff(end_tc),
        clip_start=Timecode.from_hhmmssff(start_tc),
        clip_end=Timecode.from_hhmmssff(end_tc),
        lead_in=Timecode(total_frames=0),
        tail_after=Timecode(total_frames=0),
    )


# ============================================================
# SECTION B - Resolve Review Planning Tests
# ============================================================

class ResolveReviewPlanningTests(unittest.TestCase):
    def test_source_reference_placement_spans_chapter(self) -> None:
        chapter = parse_chapter_range("00:00:30:00", "00:01:00:00")

        placement = build_source_reference_placement(chapter, track_index=1)

        self.assertEqual(
            placement,
            ResolveClipPlacement(
                start_frame=900,
                end_frame=1799,
                record_frame=0,
                track_index=1,
                media_type=1,
            ),
        )

    def test_gap_preserved_placements_keep_relative_source_offsets(self) -> None:
        chapter = parse_chapter_range("00:00:30:00", "00:02:00:00")
        clips = [
            make_clip(1, "00:00:31:00", "00:00:33:00"),
            make_clip(2, "00:00:40:10", "00:00:41:20"),
        ]

        placements = build_gap_preserved_placements(clips, chapter, track_index=2)

        self.assertEqual(placements[0].record_frame, 30)
        self.assertEqual(placements[0].start_frame, 930)
        self.assertEqual(placements[0].end_frame, 989)
        self.assertEqual(placements[1].record_frame, 310)
        self.assertEqual(placements[1].start_frame, 1210)
        self.assertEqual(placements[1].end_frame, 1249)

    def test_packed_placements_remove_internal_gaps(self) -> None:
        clips = [
            make_clip(1, "00:00:10:00", "00:00:12:00"),
            make_clip(2, "00:00:20:00", "00:00:21:15"),
        ]

        placements = build_packed_placements(clips, track_index=3)

        self.assertEqual(placements[0].record_frame, 0)
        self.assertEqual(placements[1].record_frame, clips[0].duration.total_frames)
        self.assertEqual(placements[1].track_index, 3)

    def test_candidate_clip_from_dict_rebuilds_clip_timings(self) -> None:
        clip = candidate_clip_from_dict(
            {
                "clip_index": 3,
                "source_path": "A:\\sample.mp4",
                "chapter_start": "00:00:00:00",
                "chapter_end": "00:02:00:00",
                "activity_start": "00:00:10:02",
                "activity_end": "00:00:12:05",
                "clip_start": "00:00:10:00",
                "clip_end": "00:00:12:09",
                "lead_in": {"seconds": 0, "frames": 2},
                "tail_after": {"seconds": 0, "frames": 4},
            }
        )

        self.assertEqual(clip.clip_index, 3)
        self.assertEqual(clip.source_path, "A:\\sample.mp4")
        self.assertEqual(clip.activity_start.to_hhmmssff(), "00:00:10:02")
        self.assertEqual(clip.clip_end.to_hhmmssff(), "00:00:12:09")

    def test_candidate_clip_from_dict_clamps_negative_tail_after_to_zero(self) -> None:
        clip = candidate_clip_from_dict(
            {
                "clip_index": 4,
                "source_path": "A:\\sample.mp4",
                "chapter_start": "00:00:00:00",
                "chapter_end": "00:02:00:00",
                "activity_start": "00:01:58:00",
                "activity_end": "00:02:00:01",
                "clip_start": "00:01:57:28",
                "clip_end": "00:02:00:00",
                "lead_in": {"seconds": 0, "frames": 2},
                "tail_after": {"seconds": -1, "frames": 29},
            }
        )

        self.assertEqual(clip.tail_after.total_frames, 0)
        self.assertEqual(clip.clip_end.to_hhmmssff(), "00:02:00:00")

    def test_load_review_payload_uses_first_clip_as_source_and_chapter_reference(self) -> None:
        payload = {
            "clips": [
                {
                    "clip_index": 1,
                    "source_path": "A:\\sample.mp4",
                    "chapter_start": "00:00:00:00",
                    "chapter_end": "00:02:00:00",
                    "activity_start": "00:00:10:02",
                    "activity_end": "00:00:12:05",
                    "clip_start": "00:00:10:00",
                    "clip_end": "00:00:12:09",
                    "lead_in": {"seconds": 0, "frames": 2},
                    "tail_after": {"seconds": 0, "frames": 4},
                },
                {
                    "clip_index": 2,
                    "source_path": "A:\\sample.mp4",
                    "chapter_start": "00:00:00:00",
                    "chapter_end": "00:02:00:00",
                    "activity_start": "00:00:20:00",
                    "activity_end": "00:00:21:00",
                    "clip_start": "00:00:19:28",
                    "clip_end": "00:00:21:04",
                    "lead_in": {"seconds": 0, "frames": 2},
                    "tail_after": {"seconds": 0, "frames": 4},
                },
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            payload_path = Path(tmpdir) / "review.json"
            payload_path.write_text(json.dumps(payload), encoding="utf-8")

            video_path, chapter_range, clips = load_review_payload(payload_path)

        self.assertEqual(video_path, Path("A:\\sample.mp4"))
        self.assertEqual(chapter_range.start.to_hhmmssff(), "00:00:00:00")
        self.assertEqual(chapter_range.end.to_hhmmssff(), "00:02:00:00")
        self.assertEqual(len(clips), 2)


if __name__ == "__main__":
    unittest.main()

