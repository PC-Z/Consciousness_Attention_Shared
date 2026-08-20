# ADR-008: Use Fixed V1 Cells for Literature-Inspired Context Proxies

## Status

Accepted

## Date

2026-08-17

## Context

Peterka et al. (2026) compared P5 responses with the same orientation presented in
a random-sequence control. Figure S1 split the 35 training repeats into five bins of
seven and contrasted the first and last bins. Figures 1D and 2 retained neuron
identity across context conditions and reported P5-A suppression, no predictable
P5-B enhancement, and P5-C enhancement relative to orientation-matched controls.

The current experiment has 100 AAAAB train sequences and test P5 counts of 10 A,
80 B, and 10 C. It has no random-sequence control. The neural matrix covers multiple
brain regions and contains denoised fluorescence, not the paper's deconvolved and
standardized estimated firing rate. An earlier Notebook 04 screened every condition
independently, so its A/B/C heatmaps had different cells and row orders and could not
support cross-condition comparisons.

## Decision

Use one fixed atlas population for every literature-inspired comparison. Default to
Allen primary visual cortex `VISp`; match numeric layer suffixes but keep `VISp`
separate from `VISpm`. Preserve source `cell_index` throughout all statistic caches,
tables, aligned arrays, and atlas plots.

Convert baseline and response windows with aligned calcium timestamps. The primary
response interval ends at each measured stimulus-off timestamp. Estimate one robust
baseline-SD scale per neuron from all train positions and reuse it across train and
test. This scale makes panels comparable within the session but is not `eFRstd`.
Keep the full baseline interval in extracted arrays, but crop literature-facing
figures to a configurable 0.5 seconds before onset and 0.5 seconds after measured
stimulus offset by default. Longer calcium tails remain an explicit exploratory
option rather than the paper-matched default.

For train, compare the first seven with the last seven sequences, show P1-P5 traces,
and test the directional P5-B first-greater-than-last prediction. Also report all
five position tests with Holm correction and a full P5-B trajectory using a
user-adjustable bin size.

For test, balance P5-A/B/C to the smallest trial count. Prefer P5-B trials immediately
preceding A or C catch sequences, then use a recorded random seed if subsampling is
required. Plot all conditions with identical cells, trial counts, row scaling, row
order, and color limits. Use a repeated-neuron omnibus test and Holm-adjusted paired
comparisons only as within-session descriptions.

Map the paper's qualitative predictions to three explicitly limited proxies:

- P5-A versus P4-A inside AAAAA holds orientation and sequence identity fixed but
  remains confounded by position and repetition.
- test P5-B versus late-train P5-B holds orientation and AAAAB structure fixed but
  remains confounded by phase and elapsed time. Test a pre-specified equivalence
  margin; do not infer equivalence from a non-significant difference.
- balanced P5-C versus P5-B holds position and trial count fixed but remains
  confounded by orientation identity.

Retain the inherited condition-responsive threshold only for separate descriptive
atlas masks. Use the same fixed V1 denominator in each panel and do not subtract the
selected counts across conditions.

Write these results under
`outputs/<session_id>/neural_analysis/peterka_inspired/`, separate from older neural
screen outputs.

## Alternatives Considered

### Compare independently selected A/B/C populations

Rejected for effect testing because the cells, denominators, and heatmap orders
differ across conditions. Independent masks remain useful only as descriptive QC.

### Use all recorded neurons by default

Rejected for the primary paper comparison because the source claim is specific to
V1. Whole-brain overview and an explicit `all` option remain available.

### Treat P5-C greater than P5-B as deviance detection

Rejected because C and B have different orientations. A same-orientation random
control is required for the paper's deviant-minus-control definition.

### Interpret a non-significant B difference as no enhancement

Rejected because failure to reject a difference does not establish equivalence.
A two-one-sided equivalence test with a declared margin is reported separately.

## Consequences

- Train and test figures are directly auditable at a fixed neuron identity.
- Test A/B/C trial counts and heatmap normalization no longer create visual
  denominator or row-order artifacts.
- Figure S1's first/last-seven logic is retained despite the longer 100-sequence run.
- Current p-values use neurons nested within one mouse and must not be presented as
  mouse-level biological replication.
- Motion is not yet excluded in this neural notebook. Figure S1 retained locomotion
  trials, but the main paper analyses used additional behavioral controls.
- Formal replication of Figures 1D and 2 requires random-sequence controls and a
  multi-mouse mixed-effects model.
