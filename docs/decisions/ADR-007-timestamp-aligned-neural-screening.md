# ADR-007: Use Timestamp-Aligned Neural Activity Screening

## Status

Accepted

The independent-condition comparison described below is superseded by ADR-008 for
literature-facing train/test analysis. The timestamp, matrix, atlas, and exploratory
screening decisions remain in force.

## Date

2026-08-16

## Context

The reference analysis notebook loads a denoised neurons-by-time matrix, plots the
complete activity heatmap and neuron coordinates on a CCF mask, then marks a neuron
responsive on one trial when:

`mean(response) > mean(baseline) + k * SD(baseline)`

It also requires `SD(response) > SD(baseline)` and retains neurons recurring in at
least four of eight trials. Its window indices assume a stated 4 Hz frame rate.

Current attention sessions have explicit channel-1 frame timestamps and measured
stimulus onsets. For m42, the valid source has 10,945 atlas neurons and 23,681 aligned
frames. `whole_brain_trace_denoised_loose.npy` has the matching 10,945 by 23,681
shape, but other denoise artifacts currently report 4 by 10,945 and cannot be bound
to both axes. The valid source is about 2 GB, so loading or repeatedly copying it is
not appropriate for an interactive notebook.

## Decision

Notebook 04 uses the matching `whole_brain_trace_denoised_loose.npy` through a
read-only NumPy memory map. Before analysis, its two axes must match both
`calcium_frames.parquet` and `valid_area_5.mat`. No shape inference is accepted when
neither orientation matches.

Load neuron coordinates, region identifiers, acronyms, names, and the transposed CCF
mask directly from `valid_area_5.mat`. Preserve zero-based `cell_index` as the join
key between traces and atlas rows.

Use aligned `t_session_s` values and measured stimulus onsets to create frame
intervals. Do not convert seconds to frames with an assumed sampling rate. The
initial defaults reproduce the reference ranges in seconds: baseline -2.5 to -0.5
seconds and response 0 to 6 seconds. The default event query is the critical train
P5-B item, and all windows and query fields remain user parameters.

Retain the reference per-trial amplitude rule and optional response-variability rule.
Generalize four-of-eight recurrence into two explicit thresholds: minimum responsive
trial count and minimum responsive trial fraction. Defaults are four trials and 0.5.
Compute baseline/response means, standard deviations, and eligibility once in N7;
apply adjustable screening thresholds separately in N8 so threshold sensitivity does
not rescan the source matrix.

After selection, interpolate every selected neuron and trial onto one relative-time
grid using the aligned frame timestamps. Subtract each neuron/trial baseline before
aggregation. Show three complementary diagnostics in N9: individual-neuron trial
means with population mean and SEM, trial-averaged neurons sorted by response peak
time with reference-style row min-max normalization, and chronological trials after
averaging the selected population. Save the underlying neuron-by-trial-by-time delta
array, cell and window indices, heatmap order, and derived matrices so figures remain
reproducible without another source scan.

The N9 plots reuse the same events that determined N8 selection. Treat their apparent
positive response as selection-biased descriptive QC, not independent validation.

Add a separate independent-condition inspection for the four P5 conditions that
actually exist in the current paradigm: train-B and test-A/B/C. Compute one shared
baseline/response statistics pass for interactive efficiency, but apply thresholds,
cell selection, trace averaging, and peak-time sorting independently within each
condition. Do not create train P5-A or P5-C conditions because those events do not
exist. Do not pool P1-P4 A with P5 events because position and global context would
be confounded.

The resulting panels are descriptive screens, not cross-condition effect tests.
Different panels can contain different neurons and row orders. A fixed-cell,
held-out analysis is required for A/B/C comparisons, and the current experiment
lacks the random-sequence control used by Peterka et al. to define deviance
detection. The literature mapping and exact limitations are recorded in
`docs/literature/peterka-2026-global-context-v1.md`.

For the whole-recording overview, retain every neuron. Default to evenly spaced frame
sampling across the full time axis for interactive speed, with complete-bin means as
an explicit slower option. Use the displayed matrix's first and 99th percentiles as
default color limits because the current source scale differs from the reference,
while retaining explicit limits as parameters. Screening always uses every original
frame inside each selected baseline and response window, independent of heatmap
display reduction and color scaling.

Write derived tables, cached statistics, parameters, and figures only beneath
`outputs/<session_id>/neural_analysis/`.

## Alternatives Considered

### Use the small `trace_dff.npy` artifact

Rejected because its 4 by 10,945 shape matches neither 23,681 aligned frames nor a
10,945-neuron by time representation. Accepting it would silently exchange axes and
break event alignment.

### Assume a constant 4 Hz sampling rate

Rejected because the measured m42 median interval is about 0.201 seconds. Timestamp
lookup preserves the actual frame-event relationship and works across sessions.

### Recompute all response statistics whenever a threshold changes

Rejected because a source scan takes roughly two minutes on the current Windows
workspace. The response moments do not depend on `k`, response count, or response
fraction, so separating computation from thresholding is exact and substantially
more usable.

### Treat the screening label as statistical selectivity

Rejected. The inherited rule has no null distribution, multiple-comparison control,
or mouse-level inference. It is retained only as an exploratory screen compatible
with the existing analysis logic.

## Consequences

- Matrix-axis mismatches stop early with a source-format error.
- Heatmap reduction affects visualization only, never response classification.
- Threshold-only sensitivity checks rerun from N8; window changes require N6 and N7.
- N9 makes response shape, neuronal heterogeneity, and trial stability visible, but a
  held-out or cross-validated analysis is still required to validate selectivity.
- N12 shows train-B and test-A/B/C P5 populations independently. Its selected counts,
  heatmap rows, and atlas maps must not be subtracted or interpreted as matched-cell
  condition effects.
- The default P5 baseline may contain preceding-item activity in dense AAAAB trains.
  Alternative windows must be recorded and compared before inferential use.
- Selected neurons remain traceable to trial counts, response fractions, atlas rows,
  exact parameters, and the source session.
