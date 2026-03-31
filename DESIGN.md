<!-- SECTION A START - Purpose -->
# Clip Harvesting Design

This document is the working design reference for the clip harvesting pipeline.

Its purpose is to define the terms, stages, and outputs of the system clearly enough that future code updates can be measured against a stable design.

The pipeline is intentionally staged. Earlier stages detect and group movement. Later stages decide whether that movement is relevant art activity.
<!-- SECTION A END - Purpose -->

<!-- SECTION B START - Terms -->
## Terms

`Movement Span`
A contiguous time interval where movement is detected.

`Footprint`
The set of grid coordinates touched by movement during a movement span, time slice, or sub-time slice.

`Candidate Union`
A grouping of one or more movement spans created during union construction.

`Rejected Union`
A candidate union that fails coarse before/after screening.

`Surviving Union`
A candidate union that passes coarse before/after screening and must be refined further.

`Time Slice`
A half of a surviving union inspected during refinement.

`Sub-Time Slice`
A further subdivision of an undetermined time slice.

`Valid Slice`
A time slice or sub-time slice whose footprint shows meaningful lasting change.

`Invalid Slice`
A time slice or sub-time slice whose footprint does not show meaningful lasting change.

`Undetermined Slice`
A time slice or sub-time slice whose result is too mixed or unclear and must be subdivided further.

`Clip`
Final output assembled from nearby valid slices after refinement is complete.
<!-- SECTION B END - Terms -->

<!-- SECTION C START - Stage 1 -->
## Stage 1: Movement Detection

Goal:
Detect where movement exists in the video and store it as movement spans.

Inputs:
The video timeline and the coordinate grid over the canvas region.

What happens:
The tool scans the video from start to finish.

When sustained visual change is detected, a movement span opens.

While sustained visual change continues, the movement span remains open.

When sustained inactivity is detected, the movement span closes.

For each movement span, the tool stores:
- start time
- end time
- footprint
- footprint size

Output:
A list of movement spans.

Stage 1 does not group spans into unions.

Stage 1 does not judge whether movement is meaningful art change.

Stage 1 does not create clips.
<!-- SECTION C END - Stage 1 -->

<!-- SECTION D START - Stage 2 -->
## Stage 2: Union Construction

Goal:
Group nearby movement spans into candidate unions that are worth investigating.

What happens:
Movement spans from Stage 1 are compared for temporal and spatial closeness.

Nearby spans that plausibly belong to the same movement episode are grouped into a single candidate union.

A candidate union stores:
- start time of the first movement span in the union
- end time of the last movement span in the union
- member movement spans
- union footprint
- union footprint size

Output:
A list of candidate unions.

A candidate union is only worth investigating.

It is not yet valid or rejected.
<!-- SECTION D END - Stage 2 -->

<!-- SECTION E START - Stage 3 -->
## Stage 3: Union Screening

Goal:
Quickly reject candidate unions that do not contain lasting change.

What happens:
For each candidate union, the tool compares the union footprint before the union and after the union.

If there is no meaningful lasting change inside the union footprint, the union becomes a rejected union.

If there is meaningful lasting change inside the union footprint, the union becomes a surviving union.

Output:
Two groups:
- rejected unions
- surviving unions

Stage 3 is a coarse filter.

Passing Stage 3 does not mean the whole union becomes a clip.
<!-- SECTION E END - Stage 3 -->

<!-- SECTION F START - Stage 4 -->
## Stage 4: Surviving Union Refinement

Goal:
Find which specific parts of a surviving union are actually valid.

What happens:
Each surviving union is split into two halves.

Those halves are called time slices.

Each time slice is tested by comparing its footprint before and after the time slice.

Each time slice is classified as one of:
- valid
- invalid
- undetermined

Valid slices are kept.

Invalid slices are discarded.

Undetermined slices are split in half again into sub-time slices.

This refinement process repeats until the sub-time slice becomes valid, invalid, or reaches the minimum subdivision size.

Output:
A set of valid refined slices.
<!-- SECTION F END - Stage 4 -->

<!-- SECTION G START - Stage 5 -->
## Stage 5: Final Clip Construction

Goal:
Assemble the final clips from the valid refined slices.

What happens:
Nearby valid slices are merged when they are close enough to belong to the same final clip.

Lead-in and tail-after rules may be applied here.

Maximum clip length rules may also be applied here.

Output:
The final clips that are written to the JSON payload and then assembled in DaVinci Resolve.
<!-- SECTION G END - Stage 5 -->

<!-- SECTION H START - Design Principles -->
## Design Principles

Idle time should be discarded as early as possible.

Candidate unions are containers for investigation, not proof of validity.

A surviving union should never automatically become a clip.

The system should prefer coarse rejection early and detailed inspection later.

Refinement should stop once the remaining uncertainty is below the minimum useful timing precision for clips.

For the current project, accuracy within roughly half a second is considered acceptable unless a later benchmark suggests a different threshold.
<!-- SECTION H END - Design Principles -->

<!-- SECTION I START - Next Coding Target -->
## Next Coding Target

The next implementation work should focus on making the live code follow this staged structure more explicitly.

The immediate priority is to separate:
- candidate-union creation
- union screening
- union refinement

The current code already contains pieces of these stages, but they are not yet cleanly separated.

Future code changes should move the validator toward this staged model instead of adding more one-off heuristics on top of the current flow.
<!-- SECTION I END - Next Coding Target -->
