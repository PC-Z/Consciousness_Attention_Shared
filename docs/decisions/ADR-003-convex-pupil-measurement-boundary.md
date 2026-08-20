# ADR-003: Use a Convex Pupil Measurement Boundary

## Status

Accepted

## Date

2026-08-15

## Context

ADR-002 retained the full thresholded pupil contour so every area estimate could be
traced to visible pixels. In the recorded videos, small bright specular reflections
can touch the detected pupil edge. The resulting dark-pixel contour bends around the
reflection and produces an anatomically implausible concavity, biased area, and an
irregular boundary even when the correct pupil was selected.

The pupil is not required to be a symmetric ellipse. Its visible outline can be
asymmetric because of projection and eyelid occlusion, but small reflection-driven
inward notches should not define the dilation measurement.

## Decision

Continue to use the thresholded dark contour to identify and score pupil candidates.
After selection, use that contour's convex hull as the measurement boundary. Compute
`pupil_area`, center, equivalent radius, perimeter, and saved boundary samples from
the convex hull. Keep the ellipse as a secondary summary only.

Retain `pupil_raw_contour_area`, `pupil_raw_perimeter_px`, and
`pupil_hull_correction_fraction` so the correction remains auditable. A large hull
correction is a QC warning requiring visual review; it is not evidence that the
candidate is anatomically correct.

This decision supersedes only ADR-002's use of the full raw contour as the primary
measurement boundary. Its annotation, stabilization, threshold, and interpolation
decisions remain in force.

## Alternatives Considered

### Fit an ellipse and use ellipse area

Rejected because it forces symmetry, can hide partial eyelid occlusion, and gives a
plausible-looking result even when the segmented candidate is wrong.

### Fill only enclosed bright holes

Insufficient because external-contour retrieval already ignores enclosed holes; the
observed failure occurs when a reflection creates an inward notch connected to the
boundary.

### Increase morphological closing globally

Rejected because a larger fixed kernel could merge the pupil with eyelid shadows or
other nearby dark structures, especially when pupil size changes.

## Consequences

- Small reflection notches no longer reduce pupil area or create concave boundaries.
- Real large concavities from occlusion are also bridged, so the correction fraction
  and representative-frame overlays are mandatory QC evidence.
- Newly exported behavior Parquet files contain three additional audit columns.
- Existing Parquet files must be regenerated before comparing their `pupil_area`
  directly with results produced under this decision.
