from __future__ import annotations

# ============================================================
# SECTION A - Imports And Helpers
# ============================================================

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harvesting_tool.cli import build_parser, build_settings, run
from harvesting_tool.detection import CandidateClip, Timecode


def make_clip() -> CandidateClip:
    return CandidateClip(
        clip_index=1,
        source_path='sample.mp4',
        chapter_start=Timecode.from_hhmmssff('00:00:00:00'),
        chapter_end=Timecode.from_hhmmssff('00:01:00:00'),
        activity_start=Timecode.from_hhmmssff('00:00:10:00'),
        activity_end=Timecode.from_hhmmssff('00:00:10:20'),
        clip_start=Timecode.from_hhmmssff('00:00:09:28'),
        clip_end=Timecode.from_hhmmssff('00:00:10:24'),
        lead_in=Timecode.from_hhmmssff('00:00:00:02'),
        tail_after=Timecode.from_hhmmssff('00:00:00:04'),
    )


# ============================================================
# SECTION B - CLI Routing Tests
# ============================================================


class CliRoutingTests(unittest.TestCase):
    def test_parser_accepts_use_legacy_detector_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                'sample.mp4',
                '--chapter-start', '00:00:00:00',
                '--chapter-end', '00:00:01:00',
                '--min-harvest', '00:00:05:00',
                '--max-harvest', '00:01:00:00',
                '--output-stem', 'out/sample',
                '--precomputed-movement-evidence-json', 'out/movement_evidence.json',
                '--use-legacy-detector',
                '--staged-stage3-art-state-prototype',
            ]
        )

        self.assertEqual(args.precomputed_movement_evidence_json, Path('out/movement_evidence.json'))
        self.assertTrue(args.use_legacy_detector)
        self.assertTrue(args.staged_stage3_art_state_prototype)

    def test_build_settings_uses_per_frame_default_for_v3_pipeline(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                'sample.mp4',
                '--chapter-start', '00:00:00:00',
                '--chapter-end', '00:00:01:00',
                '--min-harvest', '00:00:05:00',
                '--max-harvest', '00:01:00:00',
                '--output-stem', 'out/sample',
            ]
        )

        settings = build_settings(args)

        self.assertEqual(settings.sample_stride, 1)

    def test_build_settings_preserves_explicit_nondefault_stride_for_v3_pipeline(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                'sample.mp4',
                '--chapter-start', '00:00:00:00',
                '--chapter-end', '00:00:01:00',
                '--min-harvest', '00:00:05:00',
                '--max-harvest', '00:01:00:00',
                '--output-stem', 'out/sample',
                '--sample-stride', '2',
            ]
        )

        settings = build_settings(args)

        self.assertEqual(settings.sample_stride, 2)

    def test_run_uses_staged_detector_path_by_default(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_stem = Path(tmpdir) / 'cut_list'
            debug_stem = Path(tmpdir) / 'debug_run'
            args = parser.parse_args(
                [
                    'sample.mp4',
                    '--chapter-start', '00:00:00:00',
                    '--chapter-end', '00:00:01:00',
                    '--min-harvest', '00:00:05:00',
                    '--max-harvest', '00:01:00:00',
                    '--output-stem', str(output_stem),
                    '--debug-stem', str(debug_stem),
                    '--staged-stage3-art-state-prototype',
                ]
            )

            fake_ranges = [(300, 330)]
            fake_debug_payload = {'final_candidate_ranges': [{'range_index': 1}]}
            fake_debug_paths = {'final_candidate_ranges': debug_stem.with_name('debug_run_final_candidate_ranges.json')}

            with patch('harvesting_tool.cli.detect_staged_activity_ranges', return_value=(fake_ranges, fake_debug_payload)) as staged_detect_mock, \
                patch('harvesting_tool.cli.build_candidate_clips', return_value=[make_clip()]) as build_clips_mock, \
                patch('harvesting_tool.cli.write_cut_lists', return_value=(output_stem.with_suffix('.txt'), output_stem.with_suffix('.json'))) as write_cut_lists_mock, \
                patch('harvesting_tool.cli.write_staged_debug_artifacts', return_value=fake_debug_paths) as write_staged_debug_mock, \
                patch('harvesting_tool.cli.detect_candidate_clips') as legacy_detect_mock:
                text_path, json_path, debug_paths, review_result = run(args)

        staged_detect_mock.assert_called_once_with(
            video_path=args.video_path,
            chapter_range=unittest.mock.ANY,
            settings=unittest.mock.ANY,
            progress_callback=unittest.mock.ANY,
            status_callback=unittest.mock.ANY,
            debug_stem=args.debug_stem,
            use_stage3_art_state_prototype=True,
            precomputed_movement_evidence_path=None,
        )
        build_clips_mock.assert_called_once()
        write_cut_lists_mock.assert_called_once()
        write_staged_debug_mock.assert_called_once()
        legacy_detect_mock.assert_not_called()
        self.assertEqual(text_path, output_stem.with_suffix('.txt'))
        self.assertEqual(json_path, output_stem.with_suffix('.json'))
        self.assertEqual(debug_paths, fake_debug_paths)
        self.assertIsNone(review_result)


if __name__ == '__main__':
    unittest.main()



