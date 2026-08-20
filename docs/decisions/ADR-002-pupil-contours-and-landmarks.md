# ADR-002: Use Reviewed Pupil Contours with Optional Landmark Stabilization

## Status

Accepted

## Date

2026-08-15

## Context

The first behavior pipeline selected the largest dark contour in a rectangle and
reported an ellipse. The real camera frames show a small eye region, dark iris/pupil
intensities, fur near the eye, and specular highlights. A rectangle admits irrelevant
dark pixels, while an ellipse alone hides segmentation failures and cannot preserve
an irregular measured boundary.

The scientific target is stimulus-linked pupil dilation, so the area measurement
must be traceable back to visible frame pixels. Head-fixed recordings can still have
small translations. A learned keypoint or segmentation model may eventually be more
robust, but it requires representative labels across animals, sessions, illumination,
blinks, occlusion, and pupil sizes, followed by held-out error validation.

## Decision

Use a reviewed deterministic baseline with four separate annotations per camera:

- an eye-aperture polygon that limits segmentation;
- a reference-frame pupil polygon that seeds target center and area;
- optional named, high-contrast eye-adjacent landmarks for Lucas-Kanade optical-flow
  stabilization of the eye polygon;
- a movement polygon for frame-difference measurements.

For every frame, segment dark candidate contours inside the current eye polygon.
Rank candidates using shape, contrast, boundary contact, and continuity with the last
valid pupil. Preserve the full-contour area as the primary dilation measure, plus a
fixed-size boundary sample for visual audit. Keep ellipse center, axes, and angle as
secondary summaries and backward-compatible columns.

Calibrate an automatic intensity threshold once on the first processed frame, or save
a manually reviewed fixed threshold. Do not recompute a percentile threshold on every
frame because that would couple the measured area to a forced dark-pixel fraction and
could suppress real dilation changes.

Keep the ROI configuration schema backward compatible: old rectangle mappings remain
readable, while new mappings declare `type: polygon` and vertices. Save annotations
atomically. Preserve invalid/blink frames and only interpolate short gaps.

Do not add DeepLabCut as a mandatory dependency at this stage. Its labeling workflow
and trained pose model are a separate validated analysis path, not a drop-in result of
one ROI selection.

## Alternatives Considered

### Continue with rectangle plus largest dark ellipse

Rejected because it can select fur, eyelid shadow, or the entire eye aperture and does
not retain the observed contour.

### Track only manually selected pupil-edge points with optical flow

Rejected as the primary measure because optical flow accumulates drift over long
videos, specular highlights interrupt tracks, and pupil dilation moves the boundary
relative to local texture. Optical flow is limited to stabilizing external anchors.

### Make DeepLabCut mandatory now

Deferred because there is no reviewed multi-session training set or held-out pixel
error estimate yet. The deterministic workflow can generate audit evidence and help
identify diverse frames for a later trained model.

## Consequences

- ROI annotation takes longer but its geometry is visible and durable.
- Parquet behavior files contain list-valued boundary columns and are larger.
- `pupil_area_normalized` is the preferred dilation trace; ellipse axes are secondary.
- Confidence is an algorithmic score, not proof of anatomical correctness.
- Users must inspect representative frames from the beginning, middle, end, blinks,
  and illumination changes before accepting a camera annotation.
- A future learned model must be documented in a new ADR with label provenance,
  train/test separation, and held-out pixel/area error.

## References

- [DeepLabCut single-animal user guide](https://deeplabcut.github.io/DeepLabCut/docs/standardDeepLabCut_UserGuide)
- [DeepLabCut labeling guide](https://deeplabcut.github.io/DeepLabCut/docs/beginner-guides/labeling.html)
