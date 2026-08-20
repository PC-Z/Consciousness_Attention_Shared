# ADR-006: Static eye ROI and bounded manual pupil anchors

## Status

Accepted

## Date

2026-08-16

## Context

The first full `m42` extraction showed that a high candidate score did not guarantee
an anatomically correct pupil. Low-contrast frames could select a pupil fragment or
an eyelid-adjacent dark region. Block P11 also re-ran the detector and could therefore
display a different contour from the one saved in the behavior Parquet.

Frame-by-frame optical flow of four P5 landmarks accumulated drift over the 144,281
frame recording. The old stabilization scale had median 1.085 and 99th percentile
1.193. Even after periodic template re-anchoring reduced those values to 1.021 and
1.097, visual overlays showed the transformed eye polygon displaced from the visible
eye aperture late in the recording. The manually reviewed static polygon remained
better aligned because this preparation is head-fixed.

Classical segmentation still cannot infer a pupil edge that is absent from the
recorded pixels. Global rejection thresholds strong enough to remove every such
failure also remove many usable frames and can bias the dilation signal itself.

## Decision

- Use the manually reviewed static eye-aperture polygon for pupil segmentation by
  default. Continue to track and export P5 landmarks as diagnostic measurements, but
  only transform the detector ROI when `use_stabilized_eye_roi` is explicitly enabled.
- Generate adaptive dark-region candidates with 5 and 9 pixel closing kernels. The
  larger 13 pixel candidate is excluded because it can merge pupil and eyelid shadows.
- Separate detector success, quality acceptance, strict observed eligibility, and
  short-gap analysis eligibility into explicit columns.
- Use centered temporal QC, contour shape, boundary-contact fraction, confidence, and
  reference-center displacement. Do not reject a frame solely because its pupil area
  differs from the reference area; area change is the scientific signal.
- Make P11 plot the exact saved boundary instead of re-running the detector.
- Permit at most 20 manually reviewed pupil polygons per camera. A manual polygon
  replaces its exact frame and guides candidate selection only within 60 neighboring
  frames. Manual anchors are trusted during QC and remain auditable in `rois.yaml`.
- Exclude `pupil_review_required` observations from the strict analysis input by
  default. Interpolate only gaps allowed by `max_interpolation_gap_s`.

## Alternatives Considered

### Continue cumulative landmark stabilization

Rejected as the default because geometrically plausible optical-flow transforms still
drifted to the wrong texture. Tight transform bounds reduced but did not eliminate the
late-recording displacement.

### Reject candidates by reference-area ratio

Rejected because this would preferentially remove genuine constriction or dilation and
could create the adaptation trend that the experiment is intended to measure.

### Use a 13 pixel closing kernel

Rejected because target-frame checks showed that it often expanded a pupil candidate
into adjacent eyelid shadows.

### Require a trained segmentation model immediately

Deferred. A learned model remains appropriate when enough reviewed masks exist across
animals and illumination states. The current pre-experiment does not yet provide a
held-out labeled set for reliable validation.

## Consequences

- Existing behavior Parquet files must be regenerated before Notebook 03 uses the new
  quality semantics.
- P5 columns remain useful for diagnosing camera or head movement, but do not silently
  alter the pupil ROI.
- Strict coverage is lower than raw detector coverage. Notebook 03 must continue to
  enforce phase-level and trial-level coverage thresholds.
- Difficult intervals can be corrected without unbounded manual work, while every
  correction remains visible and reproducible.
