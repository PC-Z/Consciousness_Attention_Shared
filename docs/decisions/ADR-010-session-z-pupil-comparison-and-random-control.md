# ADR-010: Session-z P5-minus-P4 pupil comparison and random-context control

## Status

Accepted

## Date

2026-08-17

## Supersedes

ADR-004

## Context

Notebook 03 originally emphasized a fractional P5 response relative to a local
baseline. That representation is sensitive to a small denominator and is not
the closest available analogue of the pupil analysis in Westerberg et al. The
current test set is also unbalanced: predictable B trials are much more common
than A or C catch trials. A direct mean across every B trial therefore combines
condition effects with trial-count and recording-time differences.

Peterka et al. used a separate random-sequence control and compared the same
orientation under trained and random contexts. The present recordings do not
contain that control, so a same-orientation causal context comparison cannot be
reconstructed retrospectively.

## Decision

### Primary pupil measure

The primary response is:

`mean(P5 pupil_session_z - P4 pupil_session_z)` during the physical P5 stimulus.

The response window is 0-1 s for the dataset labelled 500 ms and 0-2 s for the
dataset labelled 1000 ms, following the confirmed event metadata rather than
the filename label. Because both values use the same session mean and standard
deviation, subtracting them is algebraically equivalent to standardizing the
raw P5-minus-P4 diameter difference by the session standard deviation. A second
"additive baseline correction" would duplicate this operation and will not be
reported as an independent analysis.

The fractional implementation is frozen at branch and tag
`pupil03-fractional-v1.0.0`. New outputs are written to a versioned v2 directory
so that old results remain intact.

### Train summaries

- The endpoint comparison uses original sequence numbers 1-10 and 91-100.
- Grouped traces use fixed original-number bins 1-10, 11-20, ..., 91-100.
- Invalid trials remain missing within their original bin; later valid trials
  never shift forward to fill a bin.
- The notebook does not add median, trimmed-mean, first-35-only, or separate
  full-session slope analyses.

### Current test-set primary comparison

Two comparisons are kept separate:

- A versus `B_A`: each valid AAAAA catch is paired only with original trial
  `sequence_index - 1` when that trial is a valid AAAAB sequence.
- C versus `B_C`: each valid AAAAC catch is paired under the same rule.

If the immediate predecessor is absent, invalid, or not AAAAB, the catch is
reported as unmatched and excluded. The analysis must not search farther back
for a convenient B trial. These neighboring-B comparisons reduce chronology
imbalance but remain an observational fallback, not a replacement for the
random-context control in Peterka et al.

### Sensitivity analysis

For A and C separately, original test sequence numbers are divided into fixed
blocks 1-10, 11-20, ..., 91-100. Within each block, valid B trials are sampled
without replacement to equal the number of valid catches in that block. The
sampling is repeated 500 times with a fixed seed.

The notebook reports the mean sampled effect, the empirical 2.5th and 97.5th
percentiles, and sign consistency. The percentile range is explicitly named a
"B-selection sensitivity interval". It measures dependence on which abundant
B trials are selected and must not be presented as a biological confidence
interval. Zhou et al. motivated repeated equal-count resampling, but the fixed
chronology blocks and percentile reporting are project-specific adaptations.

### Compact quality control

- Report valid counts and unmatched-neighbor reasons.
- Summarize baseline pupil, movement, and convex-hull correction by condition
  or pair; do not fit a large nuisance-covariate model in this preliminary step.
- List the five trials with the largest absolute primary response so tracking
  artifacts or outlier dominance can be inspected. Listing does not authorize
  automatic exclusion.

## Future random-sequence control

The practical experiment control contains 30 five-item sequences:

- orientations: 0, 22.5, 45, 67.5, 90, 112.5, 135, and 157.5 degrees;
- P1-P4 are randomized, with no identical adjacent orientations;
- P5 is exactly balanced: 10 A (0 degrees), 10 B (90 degrees), and 10 C
  (45 degrees);
- P4 must differ from P5;
- P1-P4 orientation counts should be approximately balanced over the run;
- timing, luminance, contrast, display geometry, and acquisition settings must
  match the trained-sequence test;
- place the control after a 2-3 minute gray-screen rest, or counterbalance run
  order across animals if the acquisition workflow permits it.

The preregistered comparisons are AAAAA P5-A versus random-context P5-A,
AAAAB P5-B versus random-context P5-B, and AAAAC P5-C versus random-context
P5-C. Each side contributes 10 trials and uses the same P5-minus-P4 session-z
metric. Once this control exists, same-orientation trained-versus-random
comparisons become primary and neighboring-B comparisons become secondary.

Peterka et al.'s closer replication uses 70 fully random five-item sequences
drawn from the same eight orientations and analyzes A, B, and C at P4/P5. The
30-sequence design above is a shorter, targeted project adaptation, not a
literal replication.

## Consequences

The new primary measure avoids unstable fractional denominators and makes the
P4-to-P5 contrast explicit. Fixed original-number groups preserve chronology in
the presence of invalid trials. Neighbor pairing and resampling improve the
current unbalanced analysis but cannot identify global-context effects without
the future same-orientation random control.

## References

- Peterka DS et al. *Global context rapidly shapes sensory responses in V1*.
  bioRxiv/PMC version: https://pmc.ncbi.nlm.nih.gov/articles/PMC12803279/
- Westerberg JA et al. *Stimulus-specific adaptation in primate visual cortex*.
  PMC version: https://pmc.ncbi.nlm.nih.gov/articles/PMC11741236/
- Zhou et al. 2025. *Nature Communications*.
  https://www.nature.com/articles/s41467-025-66714-8
