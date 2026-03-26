# ============================================================
# SECTION A - Editable Settings
# ============================================================

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(r"A:\Desktop\Crap to Sort\Automation Projects\Harvesting Tool")
HARVEST_JSON_PATH = REPO_ROOT / "Test footage" / "sample_benchmark.json"
REVIEW_TIMELINE_NAME = "First-Harvest-Review"
INCLUDE_SOURCE_TRACK = True


# ============================================================
# SECTION B - Repo Import Setup
# ============================================================

SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import importlib  # noqa: E402
import harvesting_tool.resolve_review as resolve_review  # noqa: E402

resolve_review = importlib.reload(resolve_review)
ResolveReviewOptions = resolve_review.ResolveReviewOptions
create_review_timeline = resolve_review.create_review_timeline
load_review_payload = resolve_review.load_review_payload
# ============================================================
# SECTION C - Resolve Script Entry Point
# ============================================================

def main() -> None:
    import __main__

    video_path, chapter_range, clips = load_review_payload(HARVEST_JSON_PATH)
    result = create_review_timeline(
        video_path=video_path,
        chapter_range=chapter_range,
        clips=clips,
        options=ResolveReviewOptions(
            timeline_name=REVIEW_TIMELINE_NAME,
            include_source_track=INCLUDE_SOURCE_TRACK,
        ),
        resolve_app=getattr(__main__, "resolve", None),
    )
    print(
        f"Created review timeline '{result.timeline_name}' from '{result.source_timeline_name}' "
        f"with {result.clip_count} accepted clips."
    )


if __name__ == "__main__":
    main()

