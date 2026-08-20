# Pupil analysis v2 and random-control implementation plan

Date: 2026-08-17

Status: Approved for implementation

## Scope

This document is the durable implementation and experiment-planning record for
Notebook 03. It covers the current train/test recordings and the future random
sequence control. It does not modify source recordings, existing alignment
outputs, ROI files, or archived v1 outputs.

## Version management

- Legacy version: `pupil03-fractional-v1.0.0`
- Legacy branch: `archive/pupil03-fractional-v1.0.0`
- New version: `pupil03-session-z-v2.0.0`
- New output root: `outputs/<session>/pupil_analysis_v2_session_z/`

The legacy branch stores the fractional-response Notebook 03 and is retained
for traceability only. Development continues on `main` with session-z P5-P4 as
the default.

## Train analysis

1. Load aligned eye traces and sequence metadata using real timestamps.
2. Display the complete AAAAB sequence response using session-z pupil diameter.
3. Extract P4 and P5 windows and calculate P5-minus-P4 session-z traces.
4. Assign fixed groups using the original `sequence_index`:
   1-10, 11-20, ..., 91-100.
5. Plot gray individual traces and black group means for each fixed bin.
6. Plot a trial heatmap and response-mean trend using the same fixed bins.
7. Compare original trials 1-10 with 91-100 using mean traces and SEM bands.

Invalid trials are omitted from their own fixed bins and remain visible through
per-bin sample counts. The implementation must not regroup the remaining valid
trials by occurrence order.

## Test analysis

### Primary adjacent comparisons

Create two non-overlapping comparison tables:

| Catch | Reference | Required relationship | Effect |
| --- | --- | --- | --- |
| AAAAA | `B_A` | valid AAAAB at `catch_index - 1` | A minus `B_A` |
| AAAAC | `B_C` | valid AAAAB at `catch_index - 1` | C minus `B_C` |

No fallback search is permitted. Save matched and unmatched tables, including
the reason for every unmatched catch. Plot paired mean traces with SEM bands and
paired per-trial response differences.

### Repeated downsampling sensitivity

Run separately for A and C:

1. Assign fixed original-number blocks of 10 test sequences.
2. Count valid catches in each block.
3. Sample the same number of valid AAAAB trials from that block without
   replacement.
4. Repeat 500 times with seed 20260817.
5. For each repeat, calculate catch-minus-sampled-B response and trace.
6. Report the mean effect, empirical 2.5/97.5 percentiles, and proportion of
   repeats with the same sign as the mean effect.

If a block has fewer valid B trials than valid catches, the code stops with an
explicit error rather than silently changing strata or sample size.

## Compact QC

Save one QC table containing:

- condition and pair counts;
- mean baseline pupil level;
- mean P5-minus-P4 movement;
- mean P4/P5 convex-hull correction fractions when available;
- the five trials with the largest absolute primary response.

The top-five list is for reviewing tracking and dominance by extreme trials. It
does not automatically delete or relabel any trial.

## Future random-sequence control protocol

### Recommended practical control

Generate 30 five-item sequences from eight orientations:

`0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5 degrees`.

Constraints:

- exactly 10 P5-A, 10 P5-B, and 10 P5-C sequences;
- P1-P4 randomized and approximately balanced across orientations;
- no identical adjacent items, including P4-to-P5;
- randomize final sequence order;
- retain the existing display timing for the selected 500/1000 condition;
- retain luminance, contrast, spatial frequency, monitor geometry, and all
  acquisition settings;
- include a 2-3 minute gray-screen rest after the trained test before control,
  or counterbalance control/test order across mice when feasible.

The generator should use a recorded random seed, validate every constraint, and
export both the order list and a machine-readable validation summary.

### Primary future comparisons

- trained AAAAA P5-A versus random-context P5-A;
- trained AAAAB P5-B versus random-context P5-B;
- trained AAAAC P5-C versus random-context P5-C.

Each side has 10 trials. Analyze P5-minus-P4 pupil session-z, neural response,
and movement QC with identical time windows. This isolates orientation identity
from global sequence context more directly than comparing A or C against B.

### Replication option

For a closer Peterka et al. replication, use 70 fully random five-item
sequences from the same eight orientations and analyze A/B/C occurrences at P4
and P5. This provides broader random-context sampling but lengthens the session.

## Required saved artifacts

- version/config JSON;
- train fixed-group trial metrics and plots;
- adjacent pair, unmatched catch, and paired-effect tables;
- 500-repeat selected-B records, effects, trace summary, and sensitivity plot;
- compact QC and top-five review tables;
- all figures in PNG and SVG.

## Interpretation boundary

Current adjacent-B and repeated-downsampling results are robustness checks for
existing data. They must not be described as a direct replication of Peterka's
same-orientation trained-versus-random global-context comparison. That claim is
reserved for data collected with the random-sequence control.
