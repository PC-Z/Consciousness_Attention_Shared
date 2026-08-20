# Pupil 03 Fractional v1.0.0

## Status

Archived on 2026-08-17. This version is retained for reproducibility and is not the
default analysis after the session-z v2 migration.

## Main behavior

- Uses the median pupil diameter in the pre-stimulus window as a separate baseline
  for each aligned item.
- Expresses the primary pupil trace as fractional change from that trial baseline.
- Computes and plots both fractional and session-z P5-minus-P4 traces.
- Compares the unbalanced test AAAAA, AAAAB, and AAAAC conditions directly.
- Groups valid traces consecutively, so invalid trials can shift displayed group
  boundaries away from the original sequence-number boundaries.
- Reports first/last 7 and first/last 30 train summaries.

## Reproduction

Use Git branch `archive/pupil03-fractional-v1.0.0` or tag
`pupil03-fractional-v1.0.0`. The archived notebook writes results to the legacy
`outputs/<session>/pupil_analysis` directory.
