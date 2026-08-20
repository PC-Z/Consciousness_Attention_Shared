# ADR-012: Share authorized alignment and pupil-analysis state outside Git

## Status

Accepted

## Date

2026-08-19

## Context

Collaborators need to continue from the owner's current results rather than
re-run every step from raw data. In particular, the current timestamp alignment,
reviewed eye/movement ROIs, manual pupil polygons/anchors, behavior Parquet
files, and Session-Z pupil outputs must be available for visual QC and further
method development.

These artifacts are session-specific and may include sensitive annotations, so
they should not be placed in the sanitized code repository. They also change at
a different rate from the Python source and should remain tied to a data/session
manifest.

## Decision

Share authorized session artifacts in the controlled cloud data repository,
organized by session and classified as raw, alignment, annotation, behavior,
pupil analysis, or neural analysis. Preserve the exact current annotation state,
including ROI vertices, landmarks, pupil seeds, and manual pupil anchors. Store
new annotation versions as dated author-specific files instead of overwriting
the previous file.

Each shared artifact is recorded in the metadata-only manifest with a
repository-relative path, kind, file size, SHA256, session ID, code tag, and
manifest version. The code repository contains only shape-only templates.

## Consequences

- Collaborators can inspect and improve the current pupil tracking without
  losing the owner's existing manual work.
- Raw camera videos must be shared when annotation changes are expected;
  Parquet outputs alone cannot support visual review.
- Cloud storage version history or dated files becomes part of the provenance
  record.
- The owner controls which sessions and annotation versions each collaborator
  can access.
- Large calcium matrices and neural outputs remain outside GitHub.
