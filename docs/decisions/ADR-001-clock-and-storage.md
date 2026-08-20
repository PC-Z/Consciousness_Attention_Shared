# ADR-001: Segment-aware calcium anchor and lazy source access

## Status

Accepted

## Date

2026-08-14

## Context

Each camera has its own marker file, 01 and 02 contain equivalent channel-1 and
channel-2 events, and some sessions contain short abandoned recording segments.
Calcium matrices are large, electrophysiology is continuous at 1 kHz, and the
source directories must remain read-only. Early channel-2 records can represent
video-control actions rather than formal stimulus boundaries.

## Decision

- Select the main recording segment before aligning any modality.
- Subtract the first channel-1 calcium timestamp independently in 01 and 02.
- Use 02 as the canonical stimulus/calcium/electrophysiology stream; use 01 for
  camera 01 and cross-stream QC.
- Use measured channel-2 onset/offset boundaries. Treat `500ms` and `1000ms` as
  historical labels; expected visual timing is 1.0/0.5 s and 2.0/1.0 s.
- Classify leading playback/click markers separately and identify Train/Test by
  complete 100-group templates.
- Read both behavior videos, but retain electrophysiology only from `02-0.dat`.
- Store compact indexes and source references. Read calcium and ephys windows
  lazily instead of copying source arrays.
- Resolve every write through a project-root policy and reject writes outside
  `alignment_pipeline/outputs`.

## Alternatives considered

### Use the first channel-2 record as stimulus onset

Rejected because all sessions contain a leading control/noise cluster roughly
one minute before the stable Train sequence.

### Infer events from condition filenames

Rejected because frame-level video inspection and all eight TTL records show
that the historical labels do not equal stripe duration.

### Average 01 and 02 absolute clock values

Rejected because the streams already coincide after per-segment calcium-anchor
alignment, while restarted sessions can have different local relative origins.

### Copy all source arrays into one aligned file

Rejected because it duplicates multi-gigabyte calcium and ephys recordings and
makes provenance harder to audit.

## Consequences

Downstream code must use explicit `condition_label_ms`, `expected_*`, and
`measured_*` fields. Missing or extra TTL events are visible in QC. Source files
remain immutable, and users access large data through session window readers.
