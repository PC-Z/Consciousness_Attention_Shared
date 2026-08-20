# ADR-005: Adaptive pupil tracking and explicit quality states

## Status

Accepted

## Date

2026-08-15

## Context

A fixed gray threshold of 55 detected only 1,052 of 144,281 frames in the first
reviewed recording. All detections stopped after frame 1209. This was not a plausible
blink pattern: the old landmark tracker accepted an abrupt 0.10x eye-region transform
at frame 1210 because it checked only whether optical flow returned coordinates. The
recording also contains large illumination changes, so one absolute intensity threshold
cannot cover the complete video.

Treating every unresolved frame as a blink would conflate animal state, illumination,
tracking drift, and segmentation failure. Treating every found contour as equally good
would hide low-confidence or heavily convex-corrected boundaries.

## Decision

- `PUPIL_THRESHOLD=None` means per-frame adaptive segmentation. Several local intensity
  percentiles generate candidates; shape, contrast, and the preceding accepted pupil
  select the boundary.
- Landmark stabilization requires forward tracking, optional forward-backward
  consistency, and a plausible global similarity transform. Implausible scale,
  rotation, translation, or fit residual leaves the eye ROI at its last trusted pose.
- A pupil candidate that jumps implausibly in center or area from the preceding frame is
  unresolved instead of replacing the temporal prior.
- Short unresolved gaps may be interpolated for analysis, but observed and recovered
  values remain distinguishable.
- Quality fields have separate meanings:
  - `pupil_valid`: an observed, temporally plausible boundary exists.
  - `pupil_analysis_valid`: an observed boundary or an allowed short-gap interpolation
    exists.
  - `pupil_review_required`: the observation is unresolved, low-confidence, or requires
    a large convex-hull correction.
- Block P8 previews one frame every two minutes. Each target uses one second of preceding
  frames to establish a local temporal prior. Block P11 samples unresolved and flagged
  frames after extraction.
- Manual correction is deferred. It is considered only if automatic tracking still
  leaves a small systematic residue after visual QC.

## Alternatives Considered

### One fixed threshold per video

Rejected because camera illumination changes caused complete detection loss despite a
visible pupil.

### Mark every missing boundary as a blink

Rejected because missing observations do not identify the biological cause.

### Immediately label frames for DeepLabCut or manual correction

Deferred because the deterministic adaptive method recovers the known failure without a
training set or extensive manual work. A learned model remains an option if multiple
recordings fail the documented QC gates.

## Consequences

- Full extraction is slower because each frame evaluates multiple threshold candidates.
- Existing behavior Parquet files do not acquire the new semantics retroactively and
  must be regenerated.
- High observed coverage is not sufficient by itself; periodic overlays and review flags
  remain mandatory.
- Downstream stimulus analysis prefers `pupil_analysis_valid` when present while retaining
  compatibility with older exports that contain only `pupil_valid`.
